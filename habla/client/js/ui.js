// Habla — UI rendering, interactions, vocab saving

import { $, esc, escA, escRx, state, send, toast } from './core.js';

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
  requestAnimationFrame(() => area.scrollTop = area.scrollHeight);
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
  requestAnimationFrame(() => area.scrollTop = area.scrollHeight);

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
  state.pendingTranscriptCard = null;
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
  state.pendingTranscriptCard = el;
  requestAnimationFrame(() => area.scrollTop = area.scrollHeight);
}

export function finalizeExchange(msg) {
  // Dedup: skip if we already have this exchange
  if (msg.exchange_id && state.exchanges.some(e => e.exchange_id === msg.exchange_id)) return;

  // Check for a pending transcript card (created by lockTranscript) — fill in translation
  if (state.pendingTranscriptCard && state.pendingTranscriptCard.parentNode) {
    const pending = state.pendingTranscriptCard;
    state.pendingTranscriptCard = null;

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
      }
    } else {
      // New card — upgrade the pending element
      pending.className = 'ex';
      pending.innerHTML = buildExchangeHTML(msg);
      pending._exData = msg;
      msg._el = pending;
      state.exchanges.push(msg);

      // Prune oldest
      while (state.exchanges.length > MAX_EXCHANGES) {
        const old = state.exchanges.shift();
        if (old._el && old._el.parentNode) old._el.remove();
      }
    }

    const area = $('#transcript');
    requestAnimationFrame(() => area.scrollTop = area.scrollHeight);
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
      // Scroll transcript area to keep latest translation visible
      requestAnimationFrame(() => area.scrollTop = area.scrollHeight);
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
  requestAnimationFrame(() => area.scrollTop = area.scrollHeight);
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
      ${hasNotes ? `<button class="notes-toggle" onclick="toggleNotes(this)">Notes</button>` : ''}
    </div>`;
}

window.toggleNotes = (btn) => {
  const card = btn.closest('.ex');
  if (!card) return;
  card.classList.toggle('notes-open');
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
    if (r.ok) { btn.textContent = 'Saved'; }
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

// --- Text selection save ---
function hideSelSave() {
  state.selSaveBtn.classList.remove('vis');
  state.selSaveData = null;
}

export function initUI() {
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
}
