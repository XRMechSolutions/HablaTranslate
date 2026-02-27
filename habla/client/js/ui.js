// Habla — UI rendering, interactions, vocab saving, ground truth comparison

import { $, esc, escA, escRx, state, send, toast } from './core.js';

// --- Smart auto-scroll ---
// Only auto-scroll if user is near the bottom; pause when they scroll up to read.
const SCROLL_THRESHOLD = 80; // px from bottom to consider "at bottom"

function scrollIfAtBottom() {
  const area = $('#transcript');
  if (!area || state.userScrolledUp) return;
  requestAnimationFrame(() => area.scrollTop = area.scrollHeight);
}

// --- Stale pending card cleanup ---
const PENDING_TIMEOUT_MS = 30000; // 30s before marking translation as unavailable

function expirePending(el) {
  if (!el || !el.parentNode || !el.classList.contains('pending-translation')) return;
  el.classList.remove('pending-translation');
  el.style.opacity = '0.5';
  const tgtEl = el.querySelector('.ex-tgt');
  if (tgtEl) {
    const tgtFlag = state.direction === 'es_to_en' ? '\u{1F1EC}\u{1F1E7}' : '\u{1F1EA}\u{1F1F8}';
    tgtEl.innerHTML = `${tgtFlag} <em style="color:var(--fg3)">[Translation unavailable]</em>`;
    tgtEl.style.opacity = '1';
  }
  state.pendingCards.delete(el);
}

// --- Streaming partials ---
export function ensurePartialCard() {
  if (state.partialEl) return state.partialEl;
  const area = $('#transcript');
  const ph = $('#placeholder');
  if (ph) ph.remove();

  state.partialEl = document.createElement('div');
  state.partialSrcText = '';
  state.partialTgtText = '';
  state.partialEl.className = 'ex partial-live live-card';
  state.partialEl.style.borderLeftColor = 'var(--accent)';
  state.partialEl.style.opacity = '0.8';
  state.partialEl.innerHTML = `
    <div class="ex-meta live-label">Live</div>
    <div class="ex-src partial-src" style="min-height:18px"></div>
    <div class="ex-tgt partial-tgt" style="min-height:18px;opacity:0.6"><span class="loading">Translating</span></div>
  `;
  area.appendChild(state.partialEl);
  scrollIfAtBottom();
  return state.partialEl;
}

export function showPartialSource(text) {
  const card = ensurePartialCard();
  const srcFlag = state.direction === 'es_to_en' ? '\u{1F1EA}\u{1F1F8}' : '\u{1F1EC}\u{1F1E7}';
  state.partialSrcText = text;
  const srcEl = card.querySelector('.partial-src');
  srcEl.innerHTML = `${srcFlag} ${esc(state.partialSrcText)} <span style="color:var(--accent)">\u258B</span>`;
  srcEl.scrollTop = srcEl.scrollHeight;
  const area = $('#transcript');
  if (area.lastElementChild !== card) area.appendChild(card);
  scrollIfAtBottom();

  // No watchdog — partials persist until replaced by transcript_final or new exchange.
  // User requirement: displayed content must never be auto-cleared.
}

export function showPartialTranslation(text) {
  const card = ensurePartialCard();
  const tgtFlag = state.direction === 'es_to_en' ? '\u{1F1EC}\u{1F1E7}' : '\u{1F1EA}\u{1F1F8}';
  state.partialTgtText = text;
  card.querySelector('.partial-tgt').innerHTML = `${tgtFlag} ${esc(state.partialTgtText)}`;
}

export function clearPartial() {
  clearTimeout(state.partialWatchdog);
  if (state.partialEl) { state.partialEl.remove(); state.partialEl = null; }
  state.partialSrcText = '';
  state.partialTgtText = '';
}

export function clearTranscript() {
  clearPartial();
  for (const card of state.pendingCards) clearTimeout(card._pendingTimeout);
  state.pendingCards.clear();
  state.exchanges = [];
  const area = $('#transcript');
  area.innerHTML = '';
  const ph = document.createElement('div');
  ph.className = 'placeholder';
  ph.id = 'placeholder';
  ph.innerHTML = 'Tap the microphone to start listening<br><span style="font-size:11px;color:var(--fg3)">Audio streams continuously -- no need to hold the button</span>';
  area.appendChild(ph);
}

export function lockTranscript(msg) {
  // transcript_final arrived — source text is final. Move from live to permanent card.
  clearPartial();

  const area = $('#transcript');
  const ph = $('#placeholder');
  if (ph) ph.remove();

  const srcFlag = msg.direction === 'es_to_en' ? '\u{1F1EA}\u{1F1F8}' : '\u{1F1EC}\u{1F1E7}';
  const tgtFlag = msg.direction === 'es_to_en' ? '\u{1F1EC}\u{1F1E7}' : '\u{1F1EA}\u{1F1F8}';
  const name = msg.speaker?.custom_name || msg.speaker?.label || 'Speaker';
  const color = safeColor(msg.speaker?.color);

  const el = document.createElement('div');
  el.className = 'ex pending-translation';
  el.style.borderLeftColor = color;
  el.innerHTML = `
    <div class="ex-spk">
      <div class="spk-dot" style="background:${color}"></div>
      <span class="spk-name" style="color:${color}">${esc(name)}</span>
    </div>
    <div class="ex-src">${srcFlag} ${esc(msg.text)}</div>
    <div class="ex-tgt" style="opacity:0.5">${tgtFlag} <span class="loading">Translating</span></div>
    <div class="ex-meta"><span>${new Date().toLocaleTimeString()}</span></div>
  `;

  // Store the source text on the element so finalizeExchange can match it
  el._pendingSource = msg.text;
  el._pendingSpeaker = msg.speaker;
  el._pendingDirection = msg.direction;

  area.appendChild(el);
  state.pendingCards.add(el);
  el._pendingTimeout = setTimeout(() => expirePending(el), PENDING_TIMEOUT_MS);
  scrollIfAtBottom();
}

export function finalizeExchange(msg) {
  // Dedup: skip if we already have this exchange
  if (msg.exchange_id && state.exchanges.some(e => e.exchange_id === msg.exchange_id)) return;

  // Check for a pending transcript card (created by lockTranscript) — fill in translation.
  // Pick the oldest pending card (FIFO) since translations arrive in pipeline order.
  let pending = null;
  for (const card of state.pendingCards) {
    if (card.parentNode) { pending = card; break; }
    state.pendingCards.delete(card); // stale orphan, clean up
  }
  if (pending) {
    clearTimeout(pending._pendingTimeout);
    state.pendingCards.delete(pending);

    // Use the locked source text from the pending card, not from the translation msg
    msg.source = msg.source || pending._pendingSource || '';
    msg.speaker = msg.speaker || pending._pendingSpeaker;
    msg._ts = Date.parse(msg.timestamp) || Date.now();

    // Check if we should merge with the previous exchange (same speaker, within window)
    const last = state.exchanges[state.exchanges.length - 1];
    const lastTs = last?._ts || Date.parse(last?.timestamp || '') || 0;
    const sameSpeaker = last && last.speaker?.id && msg.speaker?.id && last.speaker.id === msg.speaker.id;
    const mergeOk = sameSpeaker && (msg._ts - lastTs) < MERGE_WINDOW_MS;

    if (mergeOk) {
      // Merge into the previous card — remove the pending DOM element
      pending.remove();
      last.source = `${last.source} ${msg.source}`.trim();
      last.corrected = `${last.corrected || last.source} ${msg.corrected || msg.source}`.trim();
      last.translated = `${last.translated} ${msg.translated}`.trim();
      last.idioms = [...(last.idioms || []), ...(msg.idioms || [])];
      last.is_correction = last.is_correction || msg.is_correction;
      last.correction_detail = last.correction_detail || msg.correction_detail;
      last.confidence = msg.confidence || last.confidence;
      last.timestamp = msg.timestamp;
      last._ts = msg._ts;
      if (last._el) {
        last._el.innerHTML = buildExchangeHTML(last);
        last._el._exData = last;
        // Attach ground truth if available during playback
        if (_gtCache) attachNextGroundTruth(last._el);
      }
    } else {
      // New card — upgrade the pending element
      pending.className = 'ex';
      pending.innerHTML = buildExchangeHTML(msg);
      pending._exData = msg;
      msg._el = pending;
      state.exchanges.push(msg);

      // Attach ground truth if available during playback
      if (_gtCache) attachNextGroundTruth(pending);

      // Prune oldest
      while (state.exchanges.length > MAX_EXCHANGES) {
        const old = state.exchanges.shift();
        if (old._el && old._el.parentNode) old._el.remove();
      }
    }

    scrollIfAtBottom();
    return;
  }

  // No pending card — fall back to normal flow
  const recentPartial = state.lastPartialTime && (Date.now() - state.lastPartialTime) < 2000;

  if (recentPartial && state.partialEl) {
    // Speech is ongoing — reset partial text but keep the card for incoming speech
    state.partialSrcText = '';
    state.partialTgtText = '';
    const srcEl = state.partialEl.querySelector('.partial-src');
    const tgtEl = state.partialEl.querySelector('.partial-tgt');
    if (srcEl) srcEl.innerHTML = '';
    if (tgtEl) tgtEl.innerHTML = '<span class="loading">Listening</span>';
    // Insert final card BEFORE the live partial card
    addExchange(msg, state.partialEl);
  } else {
    clearPartial();
    addExchange(msg);
  }
}

const MERGE_WINDOW_MS = 300000; // 5 min — only speaker change breaks cards
const MAX_EXCHANGES = 200; // Prune oldest DOM nodes beyond this

function safeColor(c) {
  return /^#[0-9a-fA-F]{3,8}$/.test(c) ? c : '#536471';
}

// --- Render exchange ---
export function addExchange(msg, insertBefore = null) {
  const area = $('#transcript');
  const ph = $('#placeholder');
  if (ph) ph.remove();

  const last = state.exchanges[state.exchanges.length - 1];
  const msgTs = Date.parse(msg.timestamp) || Date.now();
  const lastTs = last?._ts || Date.parse(last?.timestamp || '') || 0;
  const sameSpeaker = last && last.speaker?.id && msg.speaker?.id && last.speaker.id === msg.speaker.id;
  const mergeOk = sameSpeaker && (msgTs - lastTs) < MERGE_WINDOW_MS;

  if (mergeOk) {
    last.source = `${last.source} ${msg.source}`.trim();
    last.corrected = `${last.corrected || last.source} ${msg.corrected || msg.source}`.trim();
    last.translated = `${last.translated} ${msg.translated}`.trim();
    last.idioms = [...(last.idioms || []), ...(msg.idioms || [])];
    last.is_correction = last.is_correction || msg.is_correction;
    last.correction_detail = last.correction_detail || msg.correction_detail;
    last.confidence = msg.confidence || last.confidence;
    last.timestamp = msg.timestamp;
    last._ts = msgTs;
    if (last._el) {
      last._el.innerHTML = buildExchangeHTML(last);
      last._el._exData = last;
      // Scroll source text to show latest within its capped area
      const srcDiv = last._el.querySelector('.ex-src:not(.corrected)');
      if (srcDiv) srcDiv.scrollTop = srcDiv.scrollHeight;
      scrollIfAtBottom();
    }
    return;
  }

  msg._ts = msgTs;
  state.exchanges.push(msg);

  // Prune oldest exchanges to prevent unbounded DOM/memory growth
  while (state.exchanges.length > MAX_EXCHANGES) {
    const old = state.exchanges.shift();
    if (old._el && old._el.parentNode) old._el.remove();
  }

  const el = document.createElement('div');
  el.className = 'ex';
  el.style.borderLeftColor = safeColor(msg.speaker?.color);
  el.innerHTML = buildExchangeHTML(msg);

  // Store exchange data on element for text selection save
  el._exData = msg;
  msg._el = el;
  if (insertBefore && insertBefore.parentNode === area) {
    area.insertBefore(el, insertBefore);
  } else {
    area.appendChild(el);
  }
  // Attach ground truth if available during playback
  if (_gtCache) attachNextGroundTruth(el);
  scrollIfAtBottom();
}

function buildExchangeHTML(msg) {
  const name = msg.speaker?.custom_name || msg.speaker?.label || 'Speaker';
  const role = msg.speaker?.role_hint ? ` \u00b7 ${msg.speaker.role_hint}` : '';
  const color = safeColor(msg.speaker?.color);
  const srcFlag = state.direction === 'es_to_en' ? '\u{1F1EA}\u{1F1F8}' : '\u{1F1EC}\u{1F1E7}';
  const tgtFlag = state.direction === 'es_to_en' ? '\u{1F1EC}\u{1F1E7}' : '\u{1F1EA}\u{1F1F8}';

  // Highlight idioms in source
  const srcRaw = msg.source || msg.corrected || '';
  let srcH = esc(srcRaw);
  (msg.idioms || []).forEach(i => {
    const e = esc(i.phrase), rx = new RegExp(escRx(e), 'gi');
    srcH = srcH.replace(rx, `<span class="hl">${e}</span>`);
  });

  const correctedLine = (msg.corrected && msg.corrected !== msg.source) ?
    `<div class="ex-src corrected" title="${escA(msg.corrected)}">~ ${esc(msg.corrected)}</div>` : '';

  let idiomH = '';
  (msg.idioms || []).forEach(i => {
    const lit = i.literal ? `<div class="idiom-lit">literally: "${esc(i.literal)}"</div>` : '';
    const reg = i.region && i.region !== 'universal' ? `<div class="idiom-reg">\u{1F4CD} ${esc(i.region)}</div>` : '';
    idiomH += `<div class="idiom" data-phrase="${escA(i.phrase)}">
      <div class="idiom-term">\u{1F4A1} ${esc(i.phrase)}</div>
      ${lit}<div class="idiom-mean">${esc(i.meaning)}</div>${reg}
      <div class="idiom-btns">
        <button class="sv" onclick="saveIdiom(this)">\u2B50 Save</button>
        <button class="dm" onclick="this.closest('.idiom').remove()">\u2715</button>
      </div>
    </div>`;
  });

  let corrH = '';
  if (msg.is_correction && msg.correction_detail) {
    const c = msg.correction_detail;
    corrH = `<div class="corr">
      <div>\u{1F4DD} Correction:</div>
      <div class="corr-wrong">\u274C ${esc(c.wrong || '')}</div>
      <div class="corr-right">\u2705 ${esc(c.right || '')}</div>
      <div class="corr-expl">${esc(c.explanation || '')}</div>
      <div class="idiom-btns"><button class="sv" onclick="saveCorr(this,'${escA(c.wrong || '')}','${escA(c.right || '')}','${escA(c.explanation || '')}')">\u2B50 Save</button></div>
    </div>`;
  }

  const hasNotes = !!(idiomH || corrH || correctedLine);
  const notesH = hasNotes ? `
    <div class="ex-notes">
      ${correctedLine}${idiomH}${corrH}
    </div>` : '';

  const audioBtn = (msg.has_audio && msg.exchange_id)
    ? `<button class="audio-play" onclick="playExAudio(this,${msg.exchange_id})" title="Play audio clip">&#9654;</button>`
    : '';

  return `
    <div class="ex-spk">
      <div class="spk-dot" style="background:${color}"></div>
      <span class="spk-name" style="color:${color}" onclick="promptRename('${escA(msg.speaker?.id || '')}')">${esc(name)}</span>
      <span class="spk-role">${esc(role)}</span>
    </div>
    <div class="ex-src" title="${escA(srcRaw)}">${srcFlag} ${srcH}</div>
    <div class="ex-tgt">${tgtFlag} ${esc(msg.translated)}</div>
    ${notesH}
    <div class="ex-meta">
      <span>${new Date(msg.timestamp).toLocaleTimeString()}</span>
      ${msg.confidence ? `<span>${Math.round(msg.confidence * 100)}%</span>` : ''}
      ${audioBtn}
      ${hasNotes ? `<button class="notes-toggle" onclick="toggleNotes(this)">Notes</button>` : ''}
    </div>`;
}

window.toggleNotes = (btn) => {
  const card = btn.closest('.ex');
  if (!card) return;
  card.classList.toggle('notes-open');
};

// --- Exchange audio playback ---
let _exAudio = null;
window.playExAudio = (btn, exchangeId) => {
  // Stop any currently playing clip
  if (_exAudio) {
    _exAudio.pause();
    _exAudio = null;
    // Reset any other playing buttons
    document.querySelectorAll('.audio-play.playing').forEach(b => {
      b.classList.remove('playing');
      b.innerHTML = '&#9654;';
    });
  }
  if (btn.classList.contains('playing')) {
    btn.classList.remove('playing');
    btn.innerHTML = '&#9654;';
    return;
  }
  const card = btn.closest('.ex');
  const exData = card?._exData;
  const sessionId = exData?.session_id || state.sessionId;
  if (!sessionId) return;
  btn.classList.add('playing');
  btn.innerHTML = '&#9632;';
  _exAudio = new Audio(`/api/sessions/${sessionId}/exchanges/${exchangeId}/audio`);
  _exAudio.onended = () => { btn.classList.remove('playing'); btn.innerHTML = '&#9654;'; _exAudio = null; };
  _exAudio.onerror = () => { btn.classList.remove('playing'); btn.innerHTML = '&#9654;'; _exAudio = null; toast('Audio unavailable', 'error'); };
  _exAudio.play().catch(() => { btn.classList.remove('playing'); btn.innerHTML = '&#9654;'; _exAudio = null; });
};

// --- Speaker updates ---
export function updateSpeakers(speakers) {
  const bar = $('#spkBar');
  if (!speakers || !speakers.length) { bar.classList.remove('vis'); return; }
  bar.classList.add('vis');
  bar.innerHTML = speakers.map(s => {
    const n = s.custom_name || s.label;
    return `<button class="spk-chip" onclick="promptRename('${escA(s.id)}')" style="border-left:3px solid ${safeColor(s.color)}">${esc(n)} <span class="cnt">(${s.utterance_count})</span></button>`;
  }).join('');
}

// --- Direction / mode UI ---
export function updateDirUI() {
  if (state.direction === 'es_to_en') {
    $('#srcLang').textContent = 'ES'; $('#tgtLang').textContent = 'EN';
  } else {
    $('#srcLang').textContent = 'EN'; $('#tgtLang').textContent = 'ES';
  }
}

export function updateModeUI() {
  const b = $('#modeBtn');
  b.textContent = state.mode;
  b.className = 'mode-btn' + (state.mode === 'classroom' ? ' classroom' : '');
}

// --- Speaker rename ---
function promptRename(id) {
  state.renamingSpeakerId = id;
  $('#renameInput').value = '';
  $('#renameModal').classList.add('vis');
  setTimeout(() => $('#renameInput').focus(), 100);
}
window.promptRename = promptRename;

// --- Vocab saving ---
async function saveIdiom(btn) {
  const card = btn.closest('.idiom'), phrase = card.dataset.phrase;
  const exCard = btn.closest('.ex');
  const ex = exCard?._exData;
  const idiom = ex?.idioms?.find(i => i.phrase === phrase);
  if (!idiom) return;
  const origText = btn.textContent;
  btn.textContent = 'Saving...'; btn.disabled = true;
  try {
    const r = await fetch('/api/vocab', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ term: idiom.phrase, literal: idiom.literal || '', meaning: idiom.meaning,
        category: idiom.type || 'idiom', source_sentence: ex.corrected || ex.source, region: idiom.region || 'universal' }) });
    if (r.ok) {
      btn.textContent = 'Saved';
      // Also add to pattern DB for future regex matching (fire-and-forget)
      fetch('/api/idioms', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phrase: idiom.phrase, meaning: idiom.meaning,
          literal: idiom.literal || '', region: idiom.region || 'universal' }) }).catch(() => {});
    }
    else { btn.textContent = origText; btn.disabled = false; toast('Failed to save idiom', 'error'); }
  } catch (e) { btn.textContent = origText; btn.disabled = false; toast('Failed to save idiom \u2014 server unreachable', 'error'); }
}
window.saveIdiom = saveIdiom;

async function saveCorr(btn, w, r, exp) {
  const origText = btn.textContent;
  btn.textContent = 'Saving...'; btn.disabled = true;
  try {
    const res = await fetch('/api/vocab', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ term: `${w} \u2192 ${r}`, meaning: exp, category: 'correction' }) });
    if (res.ok) { btn.textContent = 'Saved'; }
    else { btn.textContent = origText; btn.disabled = false; toast('Failed to save correction', 'error'); }
  } catch (e) { btn.textContent = origText; btn.disabled = false; toast('Failed to save correction \u2014 server unreachable', 'error'); }
}
window.saveCorr = saveCorr;

// --- Ground truth comparison ---

/** Cache ground truth data for the active playback recording */
let _gtCache = null;  // { recording_id, segments: [...] }
let _gtUsed = new Set();  // Track which GT segments have been matched

export function setGroundTruth(recordingId, gtData) {
  if (gtData && gtData.segments) {
    _gtCache = { recording_id: recordingId, segments: gtData.segments };
    _gtUsed = new Set();
  } else {
    _gtCache = null;
    _gtUsed = new Set();
  }
}

export function clearGroundTruth() {
  _gtCache = null;
  _gtUsed = new Set();
}

/**
 * Find the best matching ground truth segment for a pipeline output.
 * Uses word overlap similarity rather than sequential index assignment,
 * so VAD re-segmentation doesn't cause misaligned comparisons.
 */
function _findBestGTMatch(pipelineText) {
  if (!_gtCache || !pipelineText) return null;
  const pWords = new Set(pipelineText.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean));
  if (!pWords.size) return null;

  let bestIdx = -1, bestScore = 0;
  for (let i = 0; i < _gtCache.segments.length; i++) {
    if (_gtUsed.has(i)) continue;
    const gt = _gtCache.segments[i];
    const gWords = new Set((gt.transcript || '').toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(Boolean));
    if (!gWords.size) continue;
    // Jaccard similarity
    let overlap = 0;
    for (const w of pWords) { if (gWords.has(w)) overlap++; }
    const score = overlap / (pWords.size + gWords.size - overlap);
    if (score > bestScore) { bestScore = score; bestIdx = i; }
  }
  // Require at least 20% similarity to match
  if (bestIdx < 0 || bestScore < 0.2) return null;
  _gtUsed.add(bestIdx);
  return _gtCache.segments[bestIdx];
}

/**
 * Attach ground truth comparison to an exchange card.
 * Called after finalizeExchange when in playback mode.
 */
export function attachNextGroundTruth(exchangeEl) {
  if (!_gtCache) return;

  const exData = exchangeEl._exData;
  const pipelineTranscript = exData?.source || exData?.corrected || '';
  const gt = _findBestGTMatch(pipelineTranscript);
  if (!gt) return;

  const refDiv = document.createElement('div');
  refDiv.className = 'gt-ref';

  const pipelineTranslation = exData?.translated || '';

  const transcriptDiff = wordDiff(gt.transcript || '', pipelineTranscript);
  const translationDiff = wordDiff(gt.translation || '', pipelineTranslation);

  refDiv.innerHTML = `
    <div class="gt-ref-header" onclick="this.parentElement.classList.toggle('gt-open')">
      Reference (${esc(_gtCache.recording_id ? 'large-v3' : 'GT')}) - Seg ${gt.segment_id}
    </div>
    <div class="gt-ref-body">
      <div class="gt-row">
        <div class="gt-label">GT Transcript:</div>
        <div class="gt-text">${esc(gt.transcript || '')}</div>
      </div>
      <div class="gt-row">
        <div class="gt-label">Transcript Diff:</div>
        <div class="gt-diff">${transcriptDiff}</div>
      </div>
      <div class="gt-row">
        <div class="gt-label">GT Translation:</div>
        <div class="gt-text">${esc(gt.translation || '')}</div>
      </div>
      <div class="gt-row">
        <div class="gt-label">Translation Diff:</div>
        <div class="gt-diff">${translationDiff}</div>
      </div>
      ${gt.asr_corrections ? `<div class="gt-row"><div class="gt-label">ASR Notes:</div><div class="gt-text gt-note">${esc(gt.asr_corrections)}</div></div>` : ''}
    </div>
  `;

  exchangeEl.appendChild(refDiv);
}

/**
 * Simple word-level diff between two strings.
 * Returns HTML with deletions (red) and insertions (green).
 */
function wordDiff(reference, actual) {
  const refWords = reference.trim().split(/\s+/).filter(Boolean);
  const actWords = actual.trim().split(/\s+/).filter(Boolean);

  if (!refWords.length && !actWords.length) return '<span class="diff-eq">(empty)</span>';

  // LCS-based diff
  const m = refWords.length, n = actWords.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (refWords[i-1].toLowerCase() === actWords[j-1].toLowerCase()) {
        dp[i][j] = dp[i-1][j-1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
      }
    }
  }

  // Backtrack to build diff
  const parts = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && refWords[i-1].toLowerCase() === actWords[j-1].toLowerCase()) {
      parts.unshift({ type: 'eq', word: actWords[j-1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
      parts.unshift({ type: 'add', word: actWords[j-1] });
      j--;
    } else {
      parts.unshift({ type: 'del', word: refWords[i-1] });
      i--;
    }
  }

  return parts.map(p => {
    const w = esc(p.word);
    if (p.type === 'del') return `<span class="diff-del">${w}</span>`;
    if (p.type === 'add') return `<span class="diff-add">${w}</span>`;
    return `<span class="diff-eq">${w}</span>`;
  }).join(' ');
}

// --- Text selection save ---
function hideSelSave() {
  state.selSaveBtn.classList.remove('vis');
  state.selSaveData = null;
}

export function initUI() {
  // Smart auto-scroll: track when user scrolls up to read earlier content.
  // Auto-scroll resumes only when the user scrolls back near the bottom — no timer.
  state.userScrolledUp = false;
  const transcript = $('#transcript');
  const scrollBtn = $('#scrollBottom');
  transcript.addEventListener('scroll', () => {
    const atBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < SCROLL_THRESHOLD;
    state.userScrolledUp = !atBottom;
    if (scrollBtn) scrollBtn.classList.toggle('vis', state.userScrolledUp);
  });
  if (scrollBtn) {
    scrollBtn.onclick = () => {
      transcript.scrollTop = transcript.scrollHeight;
      state.userScrolledUp = false;
      scrollBtn.classList.remove('vis');
    };
  }

  // Rename modal handlers
  $('#renameCancel').onclick = () => { $('#renameModal').classList.remove('vis'); state.renamingSpeakerId = null; };
  $('#renameOk').onclick = () => {
    const n = $('#renameInput').value.trim();
    if (n && state.renamingSpeakerId) send({ type: 'rename_speaker', speaker_id: state.renamingSpeakerId, name: n });
    $('#renameModal').classList.remove('vis'); state.renamingSpeakerId = null;
  };
  $('#renameInput').onkeydown = e => { if (e.key === 'Enter') $('#renameOk').click(); };

  // Text selection save setup
  state.selSaveBtn = $('#selSave');

  document.addEventListener('selectionchange', () => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.toString().trim()) {
      hideSelSave(); return;
    }
    const anchor = sel.anchorNode, focus = sel.focusNode;
    const srcEl = anchor?.parentElement?.closest?.('.ex-src') || anchor?.closest?.('.ex-src');
    const focusSrc = focus?.parentElement?.closest?.('.ex-src') || focus?.closest?.('.ex-src');
    if (!srcEl || srcEl !== focusSrc) { hideSelSave(); return; }

    const text = sel.toString().trim();
    if (!text || text.length < 2) { hideSelSave(); return; }

    const exCard = srcEl.closest('.ex');
    const exData = exCard?._exData || null;

    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    state.selSaveBtn.style.left = `${rect.left + rect.width / 2}px`;
    state.selSaveBtn.style.top = `${rect.top - 8}px`;
    state.selSaveBtn.classList.add('vis');

    state.selSaveData = {
      term: text,
      source_sentence: exData?.corrected || exData?.source || '',
      meaning: exData?.translated || ''
    };
  });

  state.selSaveBtn.addEventListener('pointerdown', async (e) => {
    e.preventDefault();
    if (!state.selSaveData) return;
    try {
      const r = await fetch('/api/vocab', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ term: state.selSaveData.term, meaning: state.selSaveData.meaning,
          category: 'phrase', source_sentence: state.selSaveData.source_sentence }) });
      const data = await r.json();
      if (r.ok) {
        state.selSaveBtn.textContent = data.duplicate ? 'Already saved' : 'Saved!';
        setTimeout(() => { state.selSaveBtn.textContent = 'Save to Vocab'; hideSelSave(); window.getSelection()?.removeAllRanges(); }, 1200);
      }
    } catch (err) { toast('Failed to save phrase', 'error'); }
  });

  document.addEventListener('pointerdown', (e) => {
    if (e.target !== state.selSaveBtn && state.selSaveData) {
      setTimeout(hideSelSave, 200);
    }
  });

  // Long-press on translation text to copy
  let _lpTimer = null;
  transcript.addEventListener('pointerdown', (e) => {
    const tgt = e.target.closest('.ex-tgt');
    if (!tgt) return;
    _lpTimer = setTimeout(() => {
      const text = tgt.textContent.trim();
      if (text && navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => toast('Copied translation', 'info', 1500)).catch(() => {});
      }
    }, 600);
  });
  transcript.addEventListener('pointerup', () => clearTimeout(_lpTimer));
  transcript.addEventListener('pointercancel', () => clearTimeout(_lpTimer));
}
