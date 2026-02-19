// Habla — LLM provider/model settings panel

import { $, state, toast } from './core.js';

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

  loadAsrSettings();
  loadAudioSpoolSettings();
}
