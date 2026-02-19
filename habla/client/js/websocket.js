// Habla — WebSocket connection, reconnection, message dispatch

import { $, state, send, toast, showSvcNotice, hideSvcNotice,
         WS_URL, WS_BASE_DELAY, WS_MAX_DELAY, WS_MAX_ATTEMPTS,
         flushAudioSpool } from './core.js';
import { clearPartial, clearTranscript, showPartialSource, showPartialTranslation,
         lockTranscript, finalizeExchange, updateSpeakers, updateDirUI, updateModeUI } from './ui.js';

export function connect() {
  if (state.ws && (state.ws.readyState === 0 || state.ws.readyState === 1)) return;
  $('#statusDot').className = 'dot wait';
  $('#connLost').classList.remove('vis');
  state.ws = new WebSocket(WS_URL);
  state.ws.binaryType = 'arraybuffer';

  state.ws.onopen = () => {
    if (window.HABLA_DEBUG) console.log('[habla] ws open', WS_URL);
    $('#statusDot').className = 'dot ok';
    state.wsAttempt = 0;
    state.wsGaveUp = false;
    clearPartial();
    if (state.listening) {
      send({ type: 'start_listening' });
    }
    while (state.pendingTextQueue.length) {
      const msg = state.pendingTextQueue.shift();
      send(msg);
    }
    flushAudioSpool().catch(() => {});
  };

  state.ws.onclose = () => {
    if (window.HABLA_DEBUG) console.log('[habla] ws close');
    $('#statusDot').className = 'dot';
    scheduleReconnect();
  };

  state.ws.onerror = () => {
    if (window.HABLA_DEBUG) console.log('[habla] ws error');
    $('#statusDot').className = 'dot';
  };
  state.ws.onmessage = e => {
    if (window.HABLA_DEBUG) {
      try { console.log('[habla] ws msg', JSON.parse(e.data)); } catch {}
    }
    try { handleMsg(JSON.parse(e.data)); } catch (err) {}
  };
}

function scheduleReconnect() {
  if (state.wsGaveUp) return;
  state.wsAttempt++;
  if (state.wsAttempt > WS_MAX_ATTEMPTS) {
    state.wsGaveUp = true;
    $('#connLost').classList.add('vis');
    $('#connInfo').textContent = `Failed after ${WS_MAX_ATTEMPTS} attempts`;
    return;
  }
  const jitter = 0.5 + Math.random(); // 0.5x to 1.5x randomization
  const delay = Math.min(WS_BASE_DELAY * Math.pow(2, state.wsAttempt - 1) * jitter, WS_MAX_DELAY);
  const secs = Math.round(delay / 1000);
  $('#connLost').classList.add('vis');
  $('#connInfo').textContent = `Reconnecting in ${secs}s (${state.wsAttempt}/${WS_MAX_ATTEMPTS})`;
  state.wsReconnTimer = setTimeout(() => {
    $('#connLost').classList.remove('vis');
    connect();
  }, delay);
}

function handleMsg(m) {
  switch (m.type) {
    case 'status':
      state.direction = m.direction || state.direction;
      state.mode = m.mode || state.mode;
      if (m.session_id) state.sessionId = m.session_id;
      updateDirUI(); updateModeUI();
      if (m.pipeline_ready === false) showSvcNotice('ASR unavailable \u2014 text-only mode');
      else hideSvcNotice();
      break;
    case 'partial': state.lastPartialTime = Date.now(); showPartialSource(m.text); break;
    case 'partial_translation': state.lastPartialTime = Date.now(); showPartialTranslation(m.text); break;
    case 'transcript_final': lockTranscript(m); break;
    case 'translation': finalizeExchange(m); break;
    case 'speakers_updated': updateSpeakers(m.speakers); break;
    case 'direction_changed': state.direction = m.direction; updateDirUI(); break;
    case 'mode_changed': state.mode = m.mode; updateModeUI(); break;
    case 'listening_started':
      state.listening = true;
      $('#listenBtn').classList.add('active');
      $('#listenBar').classList.add('active');
      // Show recording indicator if server is recording
      if (m.recording) {
        const recInfo = $('#recordingInfo');
        if (recInfo) {
          recInfo.style.display = 'block';
          recInfo.textContent = 'Recording audio to server';
        }
      }
      break;
    case 'listening_stopped':
      state.listening = false;
      $('#listenBtn').classList.remove('active');
      $('#listenBar').classList.remove('active');
      clearInterval(state.timerInterval);
      break;
    case 'session_reset':
      state.sessionId = m.session_id;
      state.direction = m.direction || state.direction;
      state.mode = m.mode || state.mode;
      clearTranscript();
      updateDirUI(); updateModeUI();
      break;
    case 'error':
      toast(m.message || 'Server error', 'error');
      if (m.message && /timeout|unavailable|ollama/i.test(m.message))
        showSvcNotice('Translation service timeout \u2014 retrying may help');
      break;
  }
}

export function initWebSocket() {
  $('#connRetry').onclick = () => {
    clearTimeout(state.wsReconnTimer);
    state.wsAttempt = 0;
    state.wsGaveUp = false;
    $('#connLost').classList.remove('vis');
    connect();
  };
}
