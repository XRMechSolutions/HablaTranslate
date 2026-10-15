package com.xrmech.habla.app.capture

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioRecordingConfiguration
import android.media.MediaRecorder
import android.media.audiofx.NoiseSuppressor
import android.os.Handler
import android.os.Looper
import com.xrmech.habla.app.capture.MicReopenPolicy.OpenStage
import com.xrmech.habla.app.log.HLog
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * The device microphone (design section 5): VOICE_RECOGNITION, 16 kHz mono PCM16 requested
 * directly, no AEC/NS/AGC effects (parity with the server input), blocking reads. A dead record
 * (negative read code, e.g. ERROR_DEAD_OBJECT after an audioserver restart) is released and
 * reopened with backoff 250..4000 ms; after the 5th failed attempt `read` throws DEAD.
 *
 * Threading: `open`/`read` run on the capture thread; `close` may come from any thread. `close`
 * only stops the record (which unblocks a pending read); the record is released by whichever
 * side finishes last, so a release never races a native read.
 */
class MicAudioSource(
    private val context: Context,
    private val audioManager: AudioManager,
) : AudioSource {
    override val sampleRate: Int = 16_000

    private val _condition = MutableStateFlow<AudioCondition>(AudioCondition.Normal)
    override val condition: StateFlow<AudioCondition> = _condition

    private val lock = Object()
    private var record: AudioRecord? = null
    private var reading = false
    @Volatile private var closed = false
    private var loggedRouting = false
    private val mainHandler = Handler(Looper.getMainLooper())

    private val recordingCallback = object : AudioManager.AudioRecordingCallback() {
        override fun onRecordingConfigChanged(configs: MutableList<AudioRecordingConfiguration>) {
            val session = synchronized(lock) { record?.audioSessionId } ?: return
            val mine = configs.firstOrNull { it.clientAudioSessionId == session } ?: return
            val silenced = mine.isClientSilenced
            val current = _condition.value
            if (silenced && current != AudioCondition.Silenced) {
                HLog.i(HLog.CAPTURE, "client silenced by the system (another app or a call holds the mic)")
                _condition.value = AudioCondition.Silenced
            } else if (!silenced && current == AudioCondition.Silenced) {
                HLog.i(HLog.CAPTURE, "client no longer silenced")
                _condition.value = AudioCondition.Normal
            }
        }
    }

    private val routingListener = AudioRecord.OnRoutingChangedListener { r -> logRouting(r, "routing changed") }

    override fun open() {
        check(!closed) { "MicAudioSource is closed" }
        synchronized(lock) { record = createStarted() }
        audioManager.registerAudioRecordingCallback(recordingCallback, mainHandler)
    }

    override fun read(buffer: ShortArray): Int {
        var attempt = 0
        while (true) {
            val r = synchronized(lock) {
                if (closed) return finishAfterClose()
                reading = true
                record
            } ?: return -1
            val n = r.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
            synchronized(lock) {
                reading = false
                if (closed) return finishAfterClose()
            }
            if (!MicReopenPolicy.needsReopen(n)) {
                if (attempt > 0) {
                    HLog.i(HLog.CAPTURE, "microphone reopened after $attempt attempt(s)")
                    _condition.value = AudioCondition.Normal
                }
                return n
            }
            attempt = reopenAfterFailure(n, attempt)
            if (attempt < 0) return -1
        }
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            record?.let { runCatching { it.stop() } }
            if (!reading) releaseLocked()
            lock.notifyAll()
        }
        runCatching { audioManager.unregisterAudioRecordingCallback(recordingCallback) }
    }

    /**
     * Releases the failed record and reopens with backoff. Returns the attempt number that
     * succeeded, -1 when closed meanwhile; throws DEAD after the last attempt.
     */
    private fun reopenAfterFailure(code: Int, previousAttempt: Int): Int {
        HLog.w(HLog.CAPTURE, "read failed (${MicReopenPolicy.describe(code)}); reopening")
        var attempt = previousAttempt
        synchronized(lock) { releaseLocked() }
        while (true) {
            attempt++
            val delay = MicReopenPolicy.delayBefore(attempt) ?: throw AudioSourceException(
                "Microphone stopped working (error $code)",
                AudioSourceException.Kind.DEAD,
                code,
            )
            _condition.value = AudioCondition.Reopening(attempt)
            synchronized(lock) {
                if (closed) return -1
                lock.wait(delay)
                if (closed) return -1
            }
            try {
                val fresh = createStarted()
                synchronized(lock) {
                    if (closed) {
                        fresh.release()
                        return -1
                    }
                    record = fresh
                }
                return attempt
            } catch (e: AudioSourceException) {
                HLog.w(HLog.CAPTURE, "reopen attempt $attempt failed: ${e.message}")
            }
        }
    }

    // RECORD_AUDIO is checked on the first line; SecurityException is still mapped below.
    @SuppressLint("MissingPermission")
    private fun createStarted(): AudioRecord {
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            throw fail(OpenStage.PERMISSION_DENIED, "Microphone permission not granted")
        }
        val minBuffer = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (minBuffer <= 0) throw fail(OpenStage.MIN_BUFFER, "Microphone unavailable: getMinBufferSize returned $minBuffer", minBuffer)
        val r = try {
            AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .build(),
                )
                .setBufferSizeInBytes(MicReopenPolicy.bufferBytes(minBuffer))
                .build()
        } catch (e: SecurityException) {
            throw fail(OpenStage.PERMISSION_DENIED, "Microphone permission denied: ${e.message}", cause = e)
        } catch (e: UnsupportedOperationException) {
            throw fail(OpenStage.NOT_INITIALIZED, "Microphone unavailable: ${e.message}", cause = e)
        }
        if (r.state != AudioRecord.STATE_INITIALIZED) {
            r.release()
            throw fail(OpenStage.NOT_INITIALIZED, "Microphone unavailable: AudioRecord not initialized")
        }
        try {
            r.startRecording()
        } catch (e: IllegalStateException) {
            r.release()
            throw fail(OpenStage.START, "Microphone unavailable: start failed (${e.message})", cause = e)
        }
        if (r.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
            r.release()
            throw fail(OpenStage.START, "Microphone unavailable: recording did not start (another app may hold the mic)")
        }
        r.addOnRoutingChangedListener(routingListener, mainHandler)
        if (!loggedRouting) {
            loggedRouting = true
            logRouting(r, "opened")
        }
        return r
    }

    private fun logRouting(r: AudioRecord, event: String) {
        val device = r.routedDevice
        val mics = runCatching { r.activeMicrophones.size }.getOrDefault(-1)
        HLog.i(
            HLog.CAPTURE,
            "$event: device=${device?.productName ?: "none"} type=${device?.type ?: -1} activeMics=$mics " +
                "noiseSuppressorAvailable=${NoiseSuppressor.isAvailable()}",
        )
    }

    private fun finishAfterClose(): Int {
        releaseLocked()
        return -1
    }

    private fun releaseLocked() {
        val r = record ?: return
        record = null
        runCatching { r.removeOnRoutingChangedListener(routingListener) }
        runCatching { r.release() }
    }

    private fun fail(stage: OpenStage, message: String, code: Int? = null, cause: Throwable? = null) =
        AudioSourceException(message, MicReopenPolicy.kindFor(stage), code, cause)
}
