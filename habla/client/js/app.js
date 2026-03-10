// Habla — application entry point

import { $, state, send, toast, initAudioSpool, loadLanguages } from './core.js';
import { updateDirUI, updateModeUI, clearTranscript, initUI } from './ui.js';
import { startListening, stopListening, initAudio } from './audio.js';
import { initSettings, autoStartPlayback } from './settings.js';
import { connect, initWebSocket } from './websocket.js';

// Initialize all modules
if (location.search.includes('debug=1')) {
  window.HABLA_DEBUG = true;
  console.log('[habla] debug enabled');
}
initAudio();
initAudioSpool();
initUI();
initSettings();
initWebSocket();

// --- Main button handlers ---
$('#listenBtn').onclick = () => {
  if (window.HABLA_DEBUG) console.log('[habla] listenBtn click', { listening: state.listening });
  state.listening ? stopListening() : startListening();
};

$('#sendBtn').onclick = () => {
  const t = $('#txtIn').value.trim();
  if (window.HABLA_DEBUG) console.log('[habla] sendBtn click', { textLen: t.length });
  if (t) { send({ type: 'text_input', text: t, speaker_id: 'MANUAL' }); $('#txtIn').value = ''; }
};
$('#txtIn').onkeydown = e => { if (e.key === 'Enter') $('#sendBtn').click(); };

$('#dirBtn').onclick = () => {
  if (state.direction === 'es_to_en') state.direction = 'en_to_es';
  else if (state.direction === 'en_to_es') state.direction = 'auto';
  else state.direction = 'es_to_en';
  localStorage.setItem('habla_direction', state.direction);
  send({ type: 'toggle_direction', direction: state.direction }); updateDirUI();
};
$('#modeBtn').onclick = () => {
  state.mode = state.mode === 'conversation' ? 'classroom' : 'conversation';
  localStorage.setItem('habla_mode', state.mode);
  send({ type: 'set_mode', mode: state.mode }); updateModeUI();
};
$('#vocabBtn').onclick = () => location.href = '/vocab';
$('#historyBtn').onclick = () => location.href = '/history';
$('#correctionsBtn').onclick = () => location.href = '/corrections';
$('#bookmarksBtn').onclick = () => location.href = '/bookmarks';

$('#saveBtn').onclick = () => {
  if (!state.sessionId) { toast('No active session', 'error'); return; }
  $('#saveModal').classList.add('vis');
  $('#saveNotesInput').value = '';
  $('#saveNotesInput').focus();
};

$('#saveOk').onclick = async () => {
  const notes = $('#saveNotesInput').value.trim();
  $('#saveModal').classList.remove('vis');

  try {
    const r = await fetch(`/api/sessions/${state.sessionId}/save`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ notes }),
    });
    if (r.ok) toast('Transcript saved', 'ok');
    else toast('Failed to save', 'error');
  } catch { toast('Failed to save — server unreachable', 'error'); }
};

$('#saveCancel').onclick = () => {
  $('#saveModal').classList.remove('vis');
};

$('#saveNotesInput').onkeydown = (e) => {
  if (e.key === 'Enter') $('#saveOk').click();
  if (e.key === 'Escape') $('#saveCancel').click();
};

$('#clearBtn').onclick = () => {
  if (!state.exchanges.length && !state.partialEl) return;
  send({ type: 'new_session' });
  clearTranscript();
};

// Keyboard shortcut: B to bookmark the most recent exchange
document.addEventListener('keydown', (e) => {
  if (e.key !== 'b' && e.key !== 'B') return;
  const ae = document.activeElement;
  if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.isContentEditable)) return;
  const cards = document.querySelectorAll('.ex .bkmk-btn');
  if (cards.length) cards[cards.length - 1].click();
});

// Keepalive with latency tracking
setInterval(() => {
  state.lastPingSent = Date.now();
  send({ type: 'ping' });
}, 30000);

// Service worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

// Restore direction/mode from localStorage before connecting
const savedDir = localStorage.getItem('habla_direction');
if (savedDir === 'es_to_en' || savedDir === 'en_to_es' || savedDir === 'auto') {
  state.direction = savedDir;
  updateDirUI();
}
// Load language registry from server
loadLanguages();
const savedMode = localStorage.getItem('habla_mode');
if (savedMode === 'conversation' || savedMode === 'classroom') {
  state.mode = savedMode;
  updateModeUI();
}

// Connect
connect();

// Auto-replay from history page
const replayParam = new URLSearchParams(location.search).get('replay');
if (replayParam) {
  const waitForWs = setInterval(() => {
    if (state.ws && state.ws.readyState === 1 && state.sessionId) {
      clearInterval(waitForWs);
      state._onSessionReset = () => autoStartPlayback(replayParam);
      send({ type: 'new_session' });
      clearTranscript();
      // Timeout: if session_reset never arrives, start anyway
      setTimeout(() => {
        if (state._onSessionReset) { state._onSessionReset = null; autoStartPlayback(replayParam); }
      }, 5000);
    }
  }, 200);
  setTimeout(() => clearInterval(waitForWs), 10000);
}
