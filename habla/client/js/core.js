// Habla — shared foundation (DOM, state, messaging, notifications)

export const $ = s => document.querySelector(s);

export function esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

export function escA(s) {
  return (s || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

export function escRx(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Shared mutable state — ES6 module singleton, all importers share the same object
export const state = {
  ws: null,
  direction: 'es_to_en',
  mode: 'conversation',
  listening: false,
  mediaRecorder: null,
  micStream: null,
  renamingSpeakerId: null,
  exchanges: [],
  timerInterval: null,
  listenStart: 0,
  partialEl: null,
  partialSrcText: '',
  partialTgtText: '',
  lastPartialTime: 0,
  partialCleanupTimer: null,
  partialWatchdog: null,
  audioMime: '',
  audioCompat: true,
  settingsProviders: null,
  wakeLock: null,
  _startingListen: false,
  selSaveBtn: null,
  selSaveData: null,
  pendingTranscriptCard: null,
  sessionId: null,
  wsAttempt: 0,
  wsReconnTimer: null,
  wsGaveUp: false,
  pendingTextQueue: [],
  audioQueue: [],
  audioQueueBytes: 0,
  audioQueueDroppedBytes: 0,
  audioFlushInProgress: false,
  audioDb: null,
  audioDbReady: false,
  audioDbBytes: 0,
  audioDbTrimInProgress: false,
  audioDbFlushInProgress: false,
  audioDbSupported: false,
};

// WebSocket URL & reconnection constants
const WS_SCHEME = location.protocol === 'https:' ? 'wss' : 'ws';
export const WS_URL = `${WS_SCHEME}://${location.host}/ws/translate`;
export const WS_BASE_DELAY = 3000;
export const WS_MAX_DELAY = 60000;
export const WS_MAX_ATTEMPTS = 20;

// --- Send helpers ---
const _QUEUEABLE_TYPES = new Set([
  'text_input', 'toggle_direction', 'set_mode', 'rename_speaker',
]);

export function send(o) {
  if (window.HABLA_DEBUG) {
    try { console.log('[habla] send', o); } catch {}
  }
  if (state.ws?.readyState === 1) {
    state.ws.send(JSON.stringify(o));
  } else if (_QUEUEABLE_TYPES.has(o.type)) {
    state.pendingTextQueue.push(o);
  }
}

export function sendBin(d) {
  if (window.HABLA_DEBUG) {
    try { console.log('[habla] sendBin', d?.byteLength || 0); } catch {}
  }
  const wsOpen = state.ws?.readyState === 1;
  if (wsOpen && !hasAudioBacklog()) {
    state.ws.send(d);
    return;
  }
  enqueueAudioChunk(d);
}

// --- Audio spooling (offline / reconnect) ---
const AUDIO_SPOOL_DEFAULT_MINUTES = 10;
const AUDIO_SPOOL_DEFAULT_MB = 50;
const AUDIO_SPOOL_MIN_MINUTES = 1;
const AUDIO_SPOOL_MAX_MINUTES = 120;
const AUDIO_SPOOL_MIN_MB = 5;
const AUDIO_SPOOL_MAX_MB = 500;
const AUDIO_DB_NAME = 'habla_audio_spool';
const AUDIO_DB_STORE = 'chunks';
const AUDIO_DB_VERSION = 1;
let _lastDropToast = 0;

function clampInt(v, min, max, fallback) {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function getSpoolLimits() {
  const mins = clampInt(localStorage.getItem('habla_audio_spool_minutes'),
    AUDIO_SPOOL_MIN_MINUTES, AUDIO_SPOOL_MAX_MINUTES, AUDIO_SPOOL_DEFAULT_MINUTES);
  const mb = clampInt(localStorage.getItem('habla_audio_spool_mb'),
    AUDIO_SPOOL_MIN_MB, AUDIO_SPOOL_MAX_MB, AUDIO_SPOOL_DEFAULT_MB);
  return { mins, mb, maxAgeMs: mins * 60 * 1000, maxBytes: mb * 1024 * 1024 };
}

function hasAudioBacklog() {
  return state.audioQueueBytes > 0 || state.audioDbBytes > 0 || state.audioFlushInProgress || state.audioDbFlushInProgress;
}

function dropOldestMemoryBytes(targetBytes) {
  let dropped = 0;
  while (state.audioQueue.length && dropped < targetBytes) {
    const item = state.audioQueue.shift();
    dropped += item.size || 0;
    state.audioQueueBytes -= item.size || 0;
  }
  if (dropped > 0) {
    state.audioQueueDroppedBytes += dropped;
    warnDroppedAudio();
  }
}

function warnDroppedAudio() {
  const now = Date.now();
  if (now - _lastDropToast < 5000) return;
  _lastDropToast = now;
  toast('Audio backlog exceeded; dropping oldest audio', 'warn', 4000);
}

function enqueueAudioChunk(buf) {
  const size = buf?.byteLength || buf?.size || 0;
  const ts = Date.now();
  if (state.audioDbReady && state.audioDb) {
    try {
      const tx = state.audioDb.transaction(AUDIO_DB_STORE, 'readwrite');
      tx.objectStore(AUDIO_DB_STORE).add({ ts, size, data: buf });
      tx.oncomplete = () => {
        state.audioDbBytes += size;
        trimAudioDb();
      };
      tx.onerror = () => {
        fallbackEnqueueMemory(buf, size, ts);
      };
      return;
    } catch (e) {
      // fall through to memory
    }
  }
  fallbackEnqueueMemory(buf, size, ts);
}

function fallbackEnqueueMemory(buf, size, ts) {
  state.audioQueue.push({ ts, size, data: buf });
  state.audioQueueBytes += size;
  trimMemoryQueue();
}

function trimMemoryQueue() {
  const { maxAgeMs, maxBytes } = getSpoolLimits();
  const now = Date.now();
  // Drop by age
  if (maxAgeMs > 0) {
    while (state.audioQueue.length && (now - state.audioQueue[0].ts) > maxAgeMs) {
      const item = state.audioQueue.shift();
      state.audioQueueBytes -= item.size || 0;
      state.audioQueueDroppedBytes += item.size || 0;
    }
  }
  // Drop by size
  if (state.audioQueueBytes > maxBytes) {
    dropOldestMemoryBytes(state.audioQueueBytes - maxBytes);
  }
}

async function openAudioDb() {
  if (!('indexedDB' in window)) return null;
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(AUDIO_DB_NAME, AUDIO_DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(AUDIO_DB_STORE)) {
        db.createObjectStore(AUDIO_DB_STORE, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function recountAudioDbBytes() {
  if (!state.audioDb) return 0;
  return new Promise((resolve) => {
    let total = 0;
    const tx = state.audioDb.transaction(AUDIO_DB_STORE, 'readonly');
    const store = tx.objectStore(AUDIO_DB_STORE);
    const req = store.openCursor();
    req.onsuccess = e => {
      const cursor = e.target.result;
      if (!cursor) { resolve(total); return; }
      total += cursor.value?.size || 0;
      cursor.continue();
    };
    req.onerror = () => resolve(total);
  });
}

async function trimAudioDb() {
  if (!state.audioDb || state.audioDbTrimInProgress) return;
  state.audioDbTrimInProgress = true;
  const { maxAgeMs, maxBytes } = getSpoolLimits();
  const now = Date.now();
  await new Promise(resolve => {
    const tx = state.audioDb.transaction(AUDIO_DB_STORE, 'readwrite');
    const store = tx.objectStore(AUDIO_DB_STORE);
    const req = store.openCursor();
    req.onsuccess = e => {
      const cursor = e.target.result;
      if (!cursor) { resolve(); return; }
      const v = cursor.value || {};
      const tooOld = maxAgeMs > 0 && (now - (v.ts || now)) > maxAgeMs;
      const tooBig = state.audioDbBytes > maxBytes;
      if (tooOld || tooBig) {
        state.audioDbBytes -= v.size || 0;
        state.audioQueueDroppedBytes += v.size || 0;
        cursor.delete();
        warnDroppedAudio();
        cursor.continue();
      } else {
        cursor.continue();
      }
    };
    req.onerror = () => resolve();
  });
  state.audioDbTrimInProgress = false;
}

async function flushMemoryQueue() {
  if (!state.audioQueue.length) return;
  state.audioFlushInProgress = true;
  try {
    let n = 0;
    while (state.audioQueue.length && state.ws?.readyState === 1) {
      const item = state.audioQueue.shift();
      state.audioQueueBytes -= item.size || 0;
      try { state.ws.send(item.data); } catch { break; }
      n++;
      if (n % 20 === 0) await new Promise(r => setTimeout(r, 0));
    }
  } finally {
    state.audioFlushInProgress = false;
  }
}

async function flushAudioDb() {
  if (!state.audioDb || state.audioDbFlushInProgress) return;
  state.audioDbFlushInProgress = true;
  try {
    // Each iteration uses its own transaction so yielding the event loop
    // between chunks doesn't auto-commit and invalidate the cursor.
    while (state.ws?.readyState === 1) {
      const item = await new Promise(resolve => {
        const tx = state.audioDb.transaction(AUDIO_DB_STORE, 'readwrite');
        const req = tx.objectStore(AUDIO_DB_STORE).openCursor();
        req.onsuccess = e => {
          const cursor = e.target.result;
          if (!cursor) { resolve(null); return; }
          const v = cursor.value;
          cursor.delete();
          resolve(v);
        };
        req.onerror = () => resolve(null);
      });
      if (!item) break;
      state.audioDbBytes -= item.size || 0;
      try { state.ws.send(item.data); } catch { break; }
    }
  } finally {
    state.audioDbFlushInProgress = false;
  }
}

export async function initAudioSpool() {
  state.audioDbSupported = 'indexedDB' in window;
  if (!state.audioDbSupported) return;
  try {
    state.audioDb = await openAudioDb();
    state.audioDbReady = !!state.audioDb;
    state.audioDbBytes = await recountAudioDbBytes();
    await trimAudioDb();
  } catch (e) {
    state.audioDbReady = false;
    state.audioDb = null;
  }
}

export async function flushAudioSpool() {
  await flushMemoryQueue();
  await flushAudioDb();
}

// --- Toast notifications ---
export function toast(msg, type = 'error', duration = 5000) {
  const wrap = $('#toastWrap');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-msg">${esc(msg)}</span><button class="toast-close" onclick="dismissToast(this.parentElement)">&times;</button>`;
  wrap.appendChild(el);
  if (duration > 0) setTimeout(() => dismissToast(el), duration);
}

export function dismissToast(el) {
  if (!el || el.classList.contains('out')) return;
  el.classList.add('out');
  el.addEventListener('animationend', () => el.remove());
  setTimeout(() => { if (el.parentNode) el.remove(); }, 500);
}

window.dismissToast = dismissToast;

// --- Service notices ---
export function showSvcNotice(msg) {
  const el = $('#svcNotice');
  el.textContent = msg;
  el.classList.add('vis');
}

export function hideSvcNotice() {
  $('#svcNotice').classList.remove('vis');
}
