// Habla — microphone capture, codec negotiation, wake lock

import { $, state, send, sendBin, toast, showSvcNotice } from './core.js';

// Voice-reactive waveform constants
const VAD_BAR_MIN_H = 3;   // px — bar height at silence
const VAD_BAR_MAX_H = 22;  // px — bar height at full volume
// Five frequency bins spread across the voice range (~100Hz–2kHz at fftSize=32)
const VAD_BIN_PICKS = [1, 2, 3, 5, 7];

export async function startListening() {
  if (window.HABLA_DEBUG) {
    console.log('[habla] startListening called', { audioCompat: state.audioCompat, audioMime: state.audioMime });
  }
  if (state.listening || state._startingListen) return;
  if (!state.audioCompat) {
    toast('Audio recording not available in this browser', 'error', 8000);
    return;
  }
  state._startingListen = true;
  try {
    const rawStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });

    // Check if audio compression is enabled
    const compressionEnabled = localStorage.getItem('habla_audio_compression') === '1';

    let finalStream = rawStream;

    if (compressionEnabled) {
      // Apply dynamics compression via Web Audio API
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(rawStream);
      const compressor = audioCtx.createDynamicsCompressor();

      // AGC settings for quiet speech
      // These values boost quiet audio while preventing clipping
      compressor.threshold.setValueAtTime(-50, audioCtx.currentTime);  // Start compression at -50dB
      compressor.knee.setValueAtTime(40, audioCtx.currentTime);        // Smooth transition
      compressor.ratio.setValueAtTime(12, audioCtx.currentTime);       // Strong compression (12:1)
      compressor.attack.setValueAtTime(0, audioCtx.currentTime);       // Instant attack
      compressor.release.setValueAtTime(0.25, audioCtx.currentTime);   // 250ms release

      const dest = audioCtx.createMediaStreamDestination();
      source.connect(compressor);
      compressor.connect(dest);

      finalStream = dest.stream;
      state.audioContext = audioCtx;  // Save for cleanup

      if (window.HABLA_DEBUG) console.log('[habla] Audio compression enabled');
    }

    state.micStream = rawStream;  // Save raw stream for cleanup

    // Voice-reactive waveform: tap AnalyserNode off the raw stream
    try {
      const actx = new AudioContext();
      const src = actx.createMediaStreamSource(rawStream);
      const analyser = actx.createAnalyser();
      analyser.fftSize = 32;  // 16 frequency bins — plenty for 5 bars
      analyser.smoothingTimeConstant = 0.75;
      src.connect(analyser);
      state._analyserCtx = actx;
      state._analyser = analyser;

      const bars = [...document.querySelectorAll('#listenBar .bar')];
      const freqData = new Uint8Array(analyser.frequencyBinCount);

      function driveBars() {
        if (!state._analyser) return;  // nulled by stopListening — safe stop condition
        analyser.getByteFrequencyData(freqData);
        bars.forEach((bar, i) => {
          const v = freqData[VAD_BIN_PICKS[i]] / 255;
          bar.style.height = `${VAD_BAR_MIN_H + (VAD_BAR_MAX_H - VAD_BAR_MIN_H) * v}px`;
        });
        state._barRaf = requestAnimationFrame(driveBars);
      }

      $('#listenBar').classList.add('js-driven');
      state._barRaf = requestAnimationFrame(driveBars);
    } catch (_) {
      // AnalyserNode unavailable — CSS animation fallback stays active
    }

    // Use higher bitrate for better quality (especially important for quiet speech)
    const bitrate = compressionEnabled ? 128000 : 96000;  // 128kbps with compression, 96kbps without

    state.mediaRecorder = new MediaRecorder(finalStream, {
      mimeType: state.audioMime,
      audioBitsPerSecond: bitrate
    });

    state.mediaRecorder.onstart = () => {
      if (window.HABLA_DEBUG) console.log('[habla] MediaRecorder started', { bitrate, compression: compressionEnabled });
    };
    state.mediaRecorder.onerror = (e) => {
      if (window.HABLA_DEBUG) console.log('[habla] MediaRecorder error', e);
    };
    state.mediaRecorder.onstop = () => {
      if (window.HABLA_DEBUG) console.log('[habla] MediaRecorder stopped');
    };

    state.mediaRecorder.ondataavailable = e => {
      if (window.HABLA_DEBUG) console.log('[habla] dataavailable', e.data?.size || 0);
      if (e.data.size > 0) e.data.arrayBuffer().then(buf => sendBin(buf));
    };

    // Stream larger chunks to improve ffmpeg decode reliability
    state.mediaRecorder.start(1000);
    state.listening = true;

    // Tell server to start VAD processing
    send({ type: 'start_listening' });

    // UI
    $('#listenBtn').classList.add('active');
    $('#listenBar').classList.add('active');

    // Timer
    state.listenStart = Date.now();
    state.timerInterval = setInterval(() => {
      const s = Math.floor((Date.now() - state.listenStart) / 1000);
      const m = Math.floor(s / 60);
      $('#timer').textContent = `${m}:${String(s % 60).padStart(2, '0')}`;
    }, 1000);
    $('#timer').textContent = '0:00';

    // Wake lock
    requestWake();

  } catch (err) {
    toast('Microphone access required. Please allow in browser settings.', 'error', 8000);
  } finally {
    state._startingListen = false;
  }
}

export function stopListening() {
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') state.mediaRecorder.stop();
  if (state.micStream) state.micStream.getTracks().forEach(t => t.stop());
  state.micStream = null; state.mediaRecorder = null; state.listening = false;

  // Clean up audio context if it was created
  if (state.audioContext) {
    state.audioContext.close().catch(() => {});
    state.audioContext = null;
  }

  // Stop voice-reactive animation
  if (state._barRaf) { cancelAnimationFrame(state._barRaf); state._barRaf = null; }
  if (state._analyserCtx) { state._analyserCtx.close().catch(() => {}); state._analyserCtx = null; }
  state._analyser = null;
  document.querySelectorAll('#listenBar .bar').forEach(b => { b.style.height = ''; });
  $('#listenBar').classList.remove('js-driven');

  // Release wake lock to save battery
  if (state.wakeLock) {
    state.wakeLock.release().catch(() => {});
    state.wakeLock = null;
  }

  send({ type: 'stop_listening' });

  $('#listenBtn').classList.remove('active');
  $('#listenBar').classList.remove('active');
  clearInterval(state.timerInterval);
}

// --- Wake lock ---
async function requestWake() {
  try { state.wakeLock = await navigator.wakeLock.request('screen'); } catch (e) {}
}

// --- Audio compatibility check ---
function checkCompat() {
  if (!window.isSecureContext) {
    showSvcNotice('Microphone requires HTTPS. Use Tailscale MagicDNS or localhost.');
    state.audioCompat = false;
  }

  if (typeof MediaRecorder === 'undefined') {
    showSvcNotice('This browser does not support audio recording. Use Chrome or Firefox.');
    state.audioCompat = false;
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    showSvcNotice('This browser does not support microphone access. Use Chrome or Firefox.');
    state.audioCompat = false;
    return;
  }

  const codecs = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg;codecs=opus',
  ];
  state.audioMime = '';
  for (const c of codecs) {
    if (MediaRecorder.isTypeSupported(c)) { state.audioMime = c; break; }
  }
  if (!state.audioMime) {
    showSvcNotice('No supported audio codec found. Audio recording unavailable.');
    state.audioCompat = false;
  }
}

export function initAudio() {
  checkCompat();
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && state.listening) requestWake();
  });
}
