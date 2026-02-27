// Habla — WebSocket connection, reconnection, message dispatch

import { $, state, send, toast, showSvcNotice, hideSvcNotice,
         WS_URL, WS_BASE_DELAY, WS_MAX_DELAY, WS_MAX_ATTEMPTS,
         flushAudioSpool } from './core.js';
import { clearPartial, clearTranscript, showPartialSource, showPartialTranslation,
         lockTranscript, finalizeExchange, updateSpeakers, updateDirUI, updateModeUI } from './ui.js';

async function restoreSession() {
  try {
    const r = await fetch('/api/system/status');
    const d = await r.json();
    // Only restore direction/mode from server if client has no localStorage preference
    if (d.direction && !localStorage.getItem('habla_direction')) {
      state.direction = d.direction; updateDirUI();
    }
    if (d.mode && !localStorage.getItem('habla_mode')) {
      state.mode = d.mode; updateModeUI();
    }
    if (d.session_id) state.sessionId = d.session_id;
  } catch {}
}
import { handlePlaybackMessage } from './settings.js';

export function connect() {
  if (state.ws && (state.ws.readyState === 0 || state.ws.readyState === 1)) return;
  $('#statusDot').className = 'dot wait';
  const _dl0 = $('#dotLabel'); if (_dl0) _dl0.textContent = 'Connecting';
  $('#connLost').classList.remove('vis');
  state.ws = new WebSocket(WS_URL);
  state.ws.binaryType = 'arraybuffer';

  state.ws.onopen = () => {
    if (window.HABLA_DEBUG) console.log('[habla] ws open', WS_URL);
    $('#statusDot').className = 'dot ok';
    const _dl1 = $('#dotLabel'); if (_dl1) _dl1.textContent = 'Connected';
    state.wsAttempt = 0;
    state.wsGaveUp = false;
    clearPartial();
    // Sync client direction/mode to server (client localStorage is authoritative)
    const ld = localStorage.getItem('habla_direction');
    if (ld === 'es_to_en' || ld === 'en_to_es') {
      send({ type: 'toggle_direction', direction: ld });
    }
    const lm = localStorage.getItem('habla_mode');
    if (lm === 'conversation' || lm === 'classroom') {
      send({ type: 'set_mode', mode: lm });
    }
    if (state.listening) {
      send({ type: 'start_listening' });
    }
    while (state.pendingTextQueue.length) {
      const msg = state.pendingTextQueue.shift();
      send(msg);
    }
    flushAudioSpool().catch(() => {});
    restoreSession();
  };

  state.ws.onclose = () => {
    if (window.HABLA_DEBUG) console.log('[habla] ws close');
    $('#statusDot').className = 'dot';
    const _dl2 = $('#dotLabel'); if (_dl2) _dl2.textContent = 'Disconnected';
    scheduleReconnect();
  };

  state.ws.onerror = () => {
    if (window.HABLA_DEBUG) console.log('[habla] ws error');
    $('#statusDot').className = 'dot';
    const _dl3 = $('#dotLabel'); if (_dl3) _dl3.textContent = 'Disconnected';
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
      // Only accept server direction/mode if client has no localStorage preference
      if (!localStorage.getItem('habla_direction')) {
        state.direction = m.direction || state.direction;
      }
      if (!localStorage.getItem('habla_mode')) {
        state.mode = m.mode || state.mode;
      }
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
    case 'playback_started':
    case 'playback_progress':
    case 'playback_finished':
    case 'playback_stopped':
      handlePlaybackMessage(m);
      break;
    case 'pong':
      if (state.lastPingSent) {
        state.latencyMs = Date.now() - state.lastPingSent;
        const dl = $('#dotLabel');
        if (dl && state.ws?.readyState === 1) dl.textContent = `Connected ${state.latencyMs}ms`;
      }
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
