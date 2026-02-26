// Habla — LLM provider/model settings panel + playback controls

import { $, state, send, toast } from './core.js';
import { clearTranscript, setGroundTruth, clearGroundTruth } from './ui.js';

const AUDIO_SPOOL_DEFAULT_MINUTES = 10;
const AUDIO_SPOOL_DEFAULT_MB = 50;
const AUDIO_SPOOL_MIN_MINUTES = 1;
const AUDIO_SPOOL_MAX_MINUTES = 120;
const AUDIO_SPOOL_MIN_MB = 5;
const AUDIO_SPOOL_MAX_MB = 500;

function clampInt(v, min, max, fallback) {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

async function openSettings() {
  $('#settingsModal').classList.add('vis');
  await loadProviders();
  await loadCurrentLLM();
  loadQualityMetrics();
}

async function loadProviders() {
  try {
    const r = await fetch('/api/llm/providers');
    const data = await r.json();
    state.settingsProviders = data.providers || [];
    if (data.active) $('#provSelect').value = data.active;
    updateProviderStatus();
    await loadModels();
  } catch (e) { toast('Failed to load LLM providers', 'warn'); }
}

function updateProviderStatus() {
  if (!state.settingsProviders) return;
  const sel = $('#provSelect').value;
  const p = state.settingsProviders.find(x => x.name === sel);
  const dot = $('#provStatus');
  dot.className = 'set-status ' + (p ? p.status : 'unknown');
  $('#costSection').style.display = sel === 'openai' ? 'block' : 'none';
}

async function loadModels() {
  const prov = $('#provSelect').value;
  const ms = $('#modelSelect');
  const qs = $('#quickModelSelect');
  ms.innerHTML = '<option value="">Loading...</option>';
  qs.innerHTML = '<option value="">(use main model)</option>';
  try {
    const r = await fetch(`/api/llm/models?provider=${prov}`);
    const data = await r.json();
    const models = data.models || [];
    if (!models.length) { ms.innerHTML = '<option value="">(no models)</option>'; return; }
    ms.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('');
    qs.innerHTML = ['<option value="">(use main model)</option>'].concat(models.map(m => `<option value="${m}">${m}</option>`)).join('');
    const saved = localStorage.getItem(`habla_llm_${prov}_model`);
    if (saved && models.includes(saved)) ms.value = saved;
    const qsaved = localStorage.getItem(`habla_llm_${prov}_quick_model`);
    if (qsaved && models.includes(qsaved)) qs.value = qsaved;
  } catch (e) {
    ms.innerHTML = '<option value="">(unavailable)</option>';
    qs.innerHTML = '<option value="">(unavailable)</option>';
  }
}

async function loadCurrentLLM() {
  try {
    const r = await fetch('/api/llm/current');
    const data = await r.json();
    if (data.metrics) {
      const m = data.metrics;
      $('#metRequests').textContent = m.successes || 0;
      const avg = m.successes ? (m.total_latency_ms / m.successes) : 0;
      $('#metLatency').textContent = avg ? `${Math.round(avg)}ms` : '-';
      $('#metFailures').textContent = m.failures || 0;
    }
    if (data.costs) {
      const c = data.costs;
      $('#costTokens').textContent = (c.session_input_tokens + c.session_output_tokens).toLocaleString();
      $('#costUsd').textContent = c.session_cost_usd.toFixed(4);
    }
    if (data.provider) $('#provSelect').value = data.provider;
    updateProviderStatus();
    if (data.model) {
      const ms = $('#modelSelect');
      const opt = [...ms.options].find(o => o.value === data.model);
      if (opt) ms.value = data.model;
    }
    if (data.quick_model) {
      const qs = $('#quickModelSelect');
      const opt = [...qs.options].find(o => o.value === data.quick_model);
      if (opt) qs.value = data.quick_model;
    }
  } catch (e) { toast('Failed to load LLM status', 'warn'); }
}

async function loadAsrSettings() {
  try {
    const r = await fetch('/api/system/status');
    const data = await r.json();
    if (typeof data.asr_auto_language === 'boolean') {
      $('#asrAutoLang').checked = data.asr_auto_language;
    } else {
      const saved = localStorage.getItem('habla_asr_auto_language');
      if (saved !== null) $('#asrAutoLang').checked = saved === '1';
    }

    // Load recording status
    if (typeof data.recording_enabled === 'boolean') {
      $('#saveAudioToggle').checked = data.recording_enabled;
      updateRecordingInfo(data.recording_enabled);
    }

    // Load audio compression setting
    const compressionSaved = localStorage.getItem('habla_audio_compression');
    $('#audioCompression').checked = compressionSaved === '1';
  } catch (e) {}
}

function loadAudioSpoolSettings() {
  const mins = clampInt(localStorage.getItem('habla_audio_spool_minutes'),
    AUDIO_SPOOL_MIN_MINUTES, AUDIO_SPOOL_MAX_MINUTES, AUDIO_SPOOL_DEFAULT_MINUTES);
  const mb = clampInt(localStorage.getItem('habla_audio_spool_mb'),
    AUDIO_SPOOL_MIN_MB, AUDIO_SPOOL_MAX_MB, AUDIO_SPOOL_DEFAULT_MB);
  $('#audioSpoolMinutes').value = mins;
  $('#audioSpoolMb').value = mb;
}

function setAudioSpoolSettings() {
  const mins = clampInt($('#audioSpoolMinutes').value,
    AUDIO_SPOOL_MIN_MINUTES, AUDIO_SPOOL_MAX_MINUTES, AUDIO_SPOOL_DEFAULT_MINUTES);
  const mb = clampInt($('#audioSpoolMb').value,
    AUDIO_SPOOL_MIN_MB, AUDIO_SPOOL_MAX_MB, AUDIO_SPOOL_DEFAULT_MB);
  $('#audioSpoolMinutes').value = mins;
  $('#audioSpoolMb').value = mb;
  localStorage.setItem('habla_audio_spool_minutes', String(mins));
  localStorage.setItem('habla_audio_spool_mb', String(mb));
  toast(`Offline audio buffer set to ${mins} min / ${mb} MB`, 'ok', 2500);
}

async function setAsrAutoLanguage(enabled) {
  try {
    await fetch('/api/system/asr/language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ auto_language: !!enabled })
    });
    localStorage.setItem('habla_asr_auto_language', enabled ? '1' : '0');
  } catch (e) { toast('Failed to update ASR setting', 'warn'); }
}

function setAudioCompression(enabled) {
  localStorage.setItem('habla_audio_compression', enabled ? '1' : '0');
  toast(enabled ? 'Audio boost enabled' : 'Audio boost disabled', 'ok');
}

async function setRecording(enabled) {
  try {
    const r = await fetch('/api/system/recording', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !!enabled })
    });
    if (r.ok) {
      updateRecordingInfo(enabled);
      toast(enabled ? 'Recording enabled' : 'Recording disabled', 'ok');
    }
  } catch (e) { toast('Failed to update recording setting', 'warn'); }
}

function updateRecordingInfo(enabled) {
  const info = $('#recordingInfo');
  if (enabled) {
    info.style.display = 'block';
    info.textContent = 'Audio will be saved to server for testing';
  } else {
    info.style.display = 'none';
  }
}

// --- Playback ---

// Track recording metadata for audio player
let _recordingHasRaw = {};
let _recordingSegCount = {};

async function loadRecordings() {
  const sel = $('#pbRecording');
  sel.innerHTML = '<option value="">Loading...</option>';
  _recordingHasRaw = {};
  _recordingSegCount = {};
  try {
    const r = await fetch('/api/recordings');
    const data = await r.json();
    if (!data.length) {
      sel.innerHTML = '<option value="">No recordings found</option>';
      updateAudioPlayer('');
      return;
    }
    data.forEach(rec => {
      _recordingHasRaw[rec.id] = !!rec.has_raw_stream;
      _recordingSegCount[rec.id] = rec.segment_count || 0;
    });
    sel.innerHTML = '<option value="">Select recording...</option>' +
      data.map(rec => {
        const date = rec.started_at ? new Date(rec.started_at).toLocaleString() : rec.id;
        const dur = rec.total_duration_seconds ? ` (${Math.round(rec.total_duration_seconds)}s)` : '';
        const segs = rec.segment_count ? `, ${rec.segment_count} segs` : '';
        const gt = rec.has_ground_truth ? ' [GT]' : '';
        const raw = rec.has_raw_stream ? '' : ' [no raw]';
        return `<option value="${rec.id}">${date}${dur}${segs}${gt}${raw}</option>`;
      }).join('');
    updateAudioPlayer(sel.value);
  } catch (e) {
    sel.innerHTML = '<option value="">Failed to load</option>';
    updateAudioPlayer('');
  }
}

function updateAudioPlayer(recordingId) {
  const section = $('#pbAudioSection');
  const audio = $('#pbAudio');
  const segSel = $('#pbSegSelect');
  const hasRaw = recordingId && _recordingHasRaw[recordingId];
  const segCount = (recordingId && _recordingSegCount[recordingId]) || 0;

  if (!recordingId || (!hasRaw && !segCount)) {
    audio.pause();
    audio.removeAttribute('src');
    section.style.display = 'none';
    return;
  }

  // Build segment selector options
  let opts = '';
  if (hasRaw) opts += '<option value="full">Full recording</option>';
  for (let i = 1; i <= segCount; i++) {
    const num = String(i).padStart(3, '0');
    opts += `<option value="segment_${num}">Segment ${i}</option>`;
  }
  segSel.innerHTML = opts;
  section.style.display = 'block';

  // Load the first available option
  _loadSegmentAudio(recordingId, segSel.value);
}

function _loadSegmentAudio(recordingId, segValue) {
  const audio = $('#pbAudio');
  const enc = encodeURIComponent(recordingId);
  if (segValue === 'full') {
    audio.src = `/api/recordings/${enc}/audio`;
  } else {
    audio.src = `/api/recordings/${enc}/audio/${segValue}.wav`;
  }
  audio.load();
}

async function startPlayback() {
  const recording = $('#pbRecording').value;
  if (!recording) { toast('Select a recording first', 'warn'); return; }

  const speed = parseFloat($('#pbSpeed').value);
  const mode = $('#pbMode').value;

  // Clear current transcript before playback
  send({ type: 'new_session' });
  clearTranscript();

  // Load ground truth if available for this recording
  clearGroundTruth();
  try {
    const gtResp = await fetch(`/api/recordings/${encodeURIComponent(recording)}`);
    if (gtResp.ok) {
      const recData = await gtResp.json();
      if (recData.ground_truth) {
        setGroundTruth(recording, recData.ground_truth);
        toast('Ground truth loaded for comparison', 'ok', 2000);
      }
    }
  } catch (e) { /* no ground truth, that's fine */ }

  try {
    const r = await fetch('/api/playback/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recording_id: recording, speed, mode }),
    });
    const data = await r.json();
    if (!r.ok) {
      toast(data.detail || 'Playback failed', 'error');
      return;
    }
    toast(`Playback started (${speed}x, ${mode})`, 'ok');
    $('#pbPlay').style.display = 'none';
    $('#pbStop').style.display = '';
    updatePlaybackStatus('Starting playback...');
  } catch (e) {
    toast('Failed to start playback', 'error');
  }
}

async function stopPlayback() {
  try {
    await fetch('/api/playback/stop', { method: 'POST' });
    toast('Playback stopped', 'ok');
  } catch (e) {
    toast('Failed to stop playback', 'error');
  }
  playbackEnded();
}

function playbackEnded() {
  $('#pbPlay').style.display = '';
  $('#pbStop').style.display = 'none';
  updatePlaybackStatus('');
  $('#listenBtn').disabled = false;
  clearGroundTruth();
}

function updatePlaybackStatus(text) {
  const el = $('#pbStatus');
  if (text) {
    el.style.display = 'block';
    el.textContent = text;
  } else {
    el.style.display = 'none';
  }
}

/** Called from websocket.js message handler */
export function handlePlaybackMessage(m) {
  switch (m.type) {
    case 'playback_started':
      $('#pbPlay').style.display = 'none';
      $('#pbStop').style.display = '';
      updatePlaybackStatus(`Playing: ${m.total_chunks} chunks at ${m.speed}x (${m.mode})`);
      $('#listenBtn').disabled = true;
      break;
    case 'playback_progress':
      updatePlaybackStatus(`Progress: ${m.chunk_index} / ${m.total_chunks}`);
      break;
    case 'playback_finished':
      playbackEnded();
      toast(`Playback complete: ${m.chunks_processed} chunks`, 'ok');
      break;
    case 'playback_stopped':
      playbackEnded();
      break;
  }
}

async function loadQualityMetrics() {
  try {
    const r = await fetch('/api/system/metrics');
    if (!r.ok) return;
    const d = await r.json();
    $('#qConfAvg').textContent = d.confidence?.average != null ? d.confidence.average.toFixed(2) : '-';
    $('#qConfLow').textContent = d.confidence?.low_count || 0;
    $('#qCorrections').textContent = d.corrections?.total_detected || 0;
    $('#qIdiomDb').textContent = d.idioms?.pattern_db_hits || 0;
    $('#qIdiomLlm').textContent = d.idioms?.llm_hits || 0;
    const avg = d.processing?.avg_ms;
    $('#qProcAvg').textContent = avg ? `${Math.round(avg)}ms` : '-';
  } catch (e) { /* metrics unavailable */ }
}

async function switchLLM() {
  const prov = $('#provSelect').value;
  const model = $('#modelSelect').value;
  const quickModel = $('#quickModelSelect').value;
  if (!model) return;
  try {
    const r = await fetch('/api/llm/select', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: prov, model: model, quick_model: quickModel }) });
    if (r.ok) {
      localStorage.setItem(`habla_llm_${prov}_model`, model);
      localStorage.setItem('habla_llm_provider', prov);
      localStorage.setItem(`habla_llm_${prov}_quick_model`, quickModel || '');
    }
  } catch (e) { toast('Failed to switch LLM provider', 'error'); }
}

export function initSettings() {
  $('#gearBtn').onclick = openSettings;
  $('#settingsClose').onclick = () => { $('#settingsModal').classList.remove('vis'); };
  $('#refreshModels').onclick = loadProviders;
  $('#qualityRefresh').onclick = loadQualityMetrics;
  $('#asrAutoLang').onchange = (e) => { setAsrAutoLanguage(e.target.checked); };
  $('#saveAudioToggle').onchange = (e) => { setRecording(e.target.checked); };
  $('#audioCompression').onchange = (e) => { setAudioCompression(e.target.checked); };
  $('#audioSpoolMinutes').onchange = () => { setAudioSpoolSettings(); };
  $('#audioSpoolMb').onchange = () => { setAudioSpoolSettings(); };

  $('#provSelect').onchange = async () => {
    updateProviderStatus();
    await loadModels();
    await switchLLM();
  };

  $('#modelSelect').onchange = async () => { await switchLLM(); };
  $('#quickModelSelect').onchange = async () => { await switchLLM(); };

  // Restore last provider on page load
  const saved = localStorage.getItem('habla_llm_provider');
  if (saved && ['ollama', 'lmstudio', 'openai'].includes(saved)) {
    fetch('/api/llm/select', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: saved,
        model: localStorage.getItem(`habla_llm_${saved}_model`) || '',
        quick_model: localStorage.getItem(`habla_llm_${saved}_quick_model`) || '' })
    }).catch(() => {});
  }

  // Playback controls
  $('#pbRefresh').onclick = loadRecordings;
  $('#pbRecording').onchange = () => updateAudioPlayer($('#pbRecording').value);
  $('#pbSegSelect').onchange = () => {
    const rec = $('#pbRecording').value;
    if (rec) _loadSegmentAudio(rec, $('#pbSegSelect').value);
  };
  $('#pbPlay').onclick = startPlayback;
  $('#pbStop').onclick = stopPlayback;

  loadAsrSettings();
  loadAudioSpoolSettings();
  loadRecordings();
}
