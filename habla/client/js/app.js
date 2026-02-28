// Habla — application entry point

import { $, state, send, toast, initAudioSpool } from './core.js';
import { updateDirUI, updateModeUI, clearTranscript, initUI } from './ui.js';
import { startListening, stopListening, initAudio } from './audio.js';
import { initSettings } from './settings.js';
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
  state.direction = state.direction === 'es_to_en' ? 'en_to_es' : 'es_to_en';
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
if (savedDir === 'es_to_en' || savedDir === 'en_to_es') {
  state.direction = savedDir;
  updateDirUI();
}
const savedMode = localStorage.getItem('habla_mode');
if (savedMode === 'conversation' || savedMode === 'classroom') {
  state.mode = savedMode;
  updateModeUI();
}

// Connect
connect();
