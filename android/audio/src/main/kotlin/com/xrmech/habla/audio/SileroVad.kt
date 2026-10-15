package com.xrmech.habla.audio

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import java.io.FileNotFoundException
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Silero VAD v6.2.1 (ONNX) through ONNX Runtime, reproducing Silero's `OnnxWrapper.__call__`
 * (R2 Q2): each call feeds `[1, 576]` = the last 64 samples of the previous frame + the 512 new
 * samples, carries the `[2, 1, 128]` state, and returns `output[0][0]`. Calibration frames never
 * reach it (the [Segmenter] skips them, like the server), so the first call sees zero state and
 * zero context. Samples must already be normalized (`s / 32768f`, D025).
 *
 * Not thread-safe: one instance per session, called from one thread.
 *
 * @param modelBytes the ONNX model, normally [bundledModelBytes]
 * @param threads ORT intra-op threads (the server's wrapper uses 1)
 * @throws IllegalArgumentException if `threads < 1` or `modelBytes` is empty
 * @throws VadException if ONNX Runtime cannot load the model or its native library, or the model
 *   I/O is not `input,state,sr -> output,stateN`
 */
class SileroVad(modelBytes: ByteArray, threads: Int = 1) : VadModel {
    private val env: OrtEnvironment
    private val options: OrtSession.SessionOptions
    private val session: OrtSession
    private val srTensor: OnnxTensor

    // Direct native-order buffers so ORT reads them without a copy.
    private val inBuf: FloatBuffer = directFloats(CONTEXT + VAD_FRAME_SAMPLES)
    private val stateBuf: FloatBuffer = directFloats(STATE_SIZE)
    private val context = FloatArray(CONTEXT)
    private var closed = false

    /**
     * Test hook: `false` feeds the bare frame as `[1, 512]` without the 64-sample context, the
     * regression R2 measured (probabilities collapse). A zero-filled context of `[1, 576]` would not
     * be detectable: the model still scores speech correctly with it.
     */
    internal var contextEnabledForTest: Boolean = true

    init {
        require(threads >= 1) { "threads must be >= 1, was $threads" }
        require(modelBytes.isNotEmpty()) { "modelBytes is empty" }
        var opts: OrtSession.SessionOptions? = null
        var sess: OrtSession? = null
        try {
            env = OrtEnvironment.getEnvironment()
            opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setInterOpNumThreads(1)
                setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            sess = env.createSession(modelBytes, opts)
            val ins = sess.inputNames
            val outs = sess.outputNames
            if (!ins.containsAll(INPUTS) || !outs.containsAll(OUTPUTS)) {
                throw VadException("unexpected model I/O: ${ins.sorted().joinToString(",")} -> ${outs.sorted().joinToString(",")}")
            }
            srTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(SAMPLE_RATE)), LongArray(0))
            options = opts
            session = sess
        } catch (e: Throwable) {
            runCatching { sess?.close() }
            runCatching { opts?.close() }
            throw when (e) {
                is VadException -> e
                is OrtException -> VadException("Silero model failed to load: ${e.message}", e)
                is UnsatisfiedLinkError, is ExceptionInInitializerError, is NoClassDefFoundError ->
                    VadException("ONNX Runtime native library unavailable: ${e.message}", e)
                else -> e
            }
        }
    }

    override fun speechProbability(frame: FloatArray): Float {
        check(!closed) { "SileroVad is closed" }
        require(frame.size == VAD_FRAME_SAMPLES) { "frame must have $VAD_FRAME_SAMPLES samples, had ${frame.size}" }
        for (i in 0 until CONTEXT) inBuf.put(i, context[i])
        for (i in 0 until VAD_FRAME_SAMPLES) inBuf.put(CONTEXT + i, frame[i])
        // Rewind/position through java.nio.Buffer: compiled against JDK 9+ the FloatBuffer overloads
        // bind covariant descriptors (FloatBuffer.rewind()LFloatBuffer;) that older Android runtimes lack.
        (inBuf as Buffer).rewind()
        (stateBuf as Buffer).rewind()
        val inputBuf = if (contextEnabledForTest) inBuf else (inBuf as Buffer).position(CONTEXT).let { inBuf.slice() }.also { (inBuf as Buffer).rewind() }
        val shape = if (contextEnabledForTest) INPUT_SHAPE else NO_CONTEXT_SHAPE
        try {
            OnnxTensor.createTensor(env, inputBuf, shape).use { input ->
                OnnxTensor.createTensor(env, stateBuf, STATE_SHAPE).use { state ->
                    session.run(mapOf("input" to input, "state" to state, "sr" to srTensor)).use { result ->
                        val prob = (result.get("output").get() as OnnxTensor).floatBuffer.get(0)
                        val stateN = (result.get("stateN").get() as OnnxTensor).floatBuffer
                        if (stateN.remaining() != STATE_SIZE) {
                            throw VadException("stateN has ${stateN.remaining()} floats, expected $STATE_SIZE")
                        }
                        (stateBuf as Buffer).rewind()
                        stateBuf.put(stateN)
                        (stateBuf as Buffer).rewind()
                        for (i in 0 until CONTEXT) context[i] = frame[VAD_FRAME_SAMPLES - CONTEXT + i]
                        return prob
                    }
                }
            }
        } catch (e: OrtException) {
            throw VadException("Silero inference failed: ${e.message}", e)
        }
    }

    override fun reset() {
        check(!closed) { "SileroVad is closed" }
        for (i in 0 until STATE_SIZE) stateBuf.put(i, 0f)
        context.fill(0f)
    }

    override fun close() {
        if (closed) return
        closed = true
        runCatching { srTensor.close() }
        runCatching { session.close() }
        runCatching { options.close() }
    }

    companion object {
        /** SHA-256 of the pinned Silero v6.2.1 `silero_vad.onnx` (same weights as the server's JIT, R2). */
        const val MODEL_SHA256: String = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
        const val MODEL_SIZE: Int = 2_327_524

        private const val RESOURCE = "silero_vad.onnx"
        private const val CONTEXT = 64
        private const val STATE_SIZE = 2 * 1 * 128
        private const val SAMPLE_RATE = 16000L
        private val INPUT_SHAPE = longArrayOf(1, (CONTEXT + VAD_FRAME_SAMPLES).toLong())
        private val NO_CONTEXT_SHAPE = longArrayOf(1, VAD_FRAME_SAMPLES.toLong())
        private val STATE_SHAPE = longArrayOf(2, 1, 128)
        private val INPUTS = setOf("input", "state", "sr")
        private val OUTPUTS = setOf("output", "stateN")

        /**
         * Reads the bundled model resource `com/xrmech/habla/audio/silero_vad.onnx`. Does not hash
         * it; callers that need a check compare against [MODEL_SHA256].
         *
         * @throws FileNotFoundException if the resource is not on the classpath
         */
        fun bundledModelBytes(): ByteArray =
            SileroVad::class.java.getResourceAsStream(RESOURCE)?.use { it.readBytes() }
                ?: throw FileNotFoundException("resource com/xrmech/habla/audio/$RESOURCE not on the classpath")

        private fun directFloats(n: Int): FloatBuffer =
            ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    }
}
