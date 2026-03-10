# Deployment

The system is designed for self-hosted deployment on a Windows laptop with an NVIDIA GPU, accessed remotely from an Android phone over Tailscale VPN.

## Deployment Topology

```
Android Phone (Chrome PWA)
  │
  │ Tailscale VPN (~20MB/hr)
  │
MSI Laptop (Windows 11, RTX 3060 12GB)
  ├── uvicorn (FastAPI, port 8002)
  ├── Ollama (port 11434, GPU)
  ├── WhisperX (GPU, in-process)
  ├── Pyannote (CPU, in-process)
  ├── Silero VAD (CPU, in-process)
  └── SQLite (data/habla.db)
```

The phone captures Opus audio and streams it over WebSocket. The server runs all ML models and returns JSON results. Bandwidth is ~20MB/hour — works on metered mobile data.

## VRAM Budget

Target hardware has known thermal and power delivery problems. The system is designed conservatively at ~5GB steady state (~42% of 12GB) to avoid crashes. The headroom is intentional.

| Component | VRAM | Loaded |
|-----------|------|--------|
| WhisperX Small | ~1GB | Always |
| Qwen3 4B Q3_K_M (Ollama) | ~2.5GB | Always |
| Pyannote 3.1 | 0 (CPU) | Always |
| Silero VAD | 0 (CPU) | Always |
| KV cache + overhead | ~1.5GB | Runtime |
| **Total** | **~5GB** | — |

All models stay loaded simultaneously. No sequential load/unload. This gives instant response but limits model sizes.

## Startup Sequence

`main.py` lifespan startup, in order:

1. **Config loading** — Environment variables, `.env` file, Pydantic defaults
2. **Logging setup** — Console handler + rotating file (10MB x 5) + error-only file (5MB x 3)
3. **Database init** — WAL mode, FTS5 triggers, integrity check, migrations
4. **Pre-pipeline health checks** — Ollama, ffmpeg, HF_TOKEN (degraded mode on failure)
5. **LLM settings restoration** — Load persisted provider/model/keys from `data/llm_settings.json`
6. **Platform guard** — Fall back from LM Studio to Ollama on non-Windows
7. **LM Studio startup** (if provider) — Launch process, load model, start monitor
8. **Pipeline creation** — PipelineOrchestrator with AudioProcessor + ContextManager
9. **TTS service** (if enabled) — Load Piper voice models
10. **Playback service** — Enumerate available recordings
11. **Pipeline startup** — Load WhisperX (GPU), Pyannote, initialize LLM
12. **Cost loading** — Restore cumulative cloud costs from DB
13. **Post-pipeline checks** — Verify WhisperX and Pyannote loaded
14. **Signal handlers** — SIGINT, SIGTERM for graceful shutdown
15. **Console command loop** — `restart`, `reload`, `quit` from stdin
16. **LLM health monitor** — Background task, checks active provider every 60s

### Shutdown

1. Cancel console task and health monitor
2. Pipeline shutdown: drain queue (30s max), cancel in-flight tasks (15s max), save session state
3. Stop LM Studio monitor and process (if applicable)
4. Close database

## Docker

`Dockerfile` (Python 3.11-slim):
- Installs ffmpeg, git, curl
- Non-root user (`habla`)
- Health check on `/health` (30s interval, 120s start period)
- Exposes port 8002

`docker-compose.yml` (two services):

| Service | Image | Port | Memory | GPU |
|---------|-------|------|--------|-----|
| `habla` | Local build | 8002 | 8GB | Yes |
| `ollama` | ollama:0.5 | 11434 | 6GB | Yes |

Habla waits for Ollama health (`/api/tags`) before starting. Ollama model data persisted in a named volume.

### Volumes

- `./data:/app/data` — Database, recordings, settings
- `./client:/app/client` — PWA files
- `ollama_data:/root/.ollama` — Model cache

## Configuration

Hierarchical Pydantic settings with env var overrides. Full list:

### Required

| Env Var | Purpose |
|---------|---------|
| `HF_TOKEN` | Pyannote speaker diarization (Hugging Face token) |

### LLM Providers

| Env Var | Default | Purpose |
|---------|---------|---------|
| `LLM_PROVIDER` | `lmstudio` | Active provider (ollama/lmstudio/openai/claude) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3:4b` | Ollama model |
| `LMSTUDIO_URL` | `http://localhost:1234` | LM Studio endpoint |
| `LMSTUDIO_MODEL` | (auto-detected) | LM Studio model |
| `LMSTUDIO_EXECUTABLE` | `lms` | LM Studio CLI path |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-nano` | OpenAI model |
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model |
| `QUICK_MODEL` | — | Override for fast partial translations |
| `RATE_LIMIT_INTERVAL` | `0.5` | Minimum seconds between LLM requests |

### ASR & Audio

| Env Var | Default | Purpose |
|---------|---------|---------|
| `WHISPER_MODEL` | `small` | WhisperX model size (tiny/base/small/medium/large-v2) |
| `WHISPER_DEVICE` | `cuda` | ASR compute device |
| `ASR_AUTO_LANGUAGE` | `true` | Auto-detect language (vs forced) |
| `VAD_THRESHOLD` | `0.35` | Silero VAD speech probability threshold |
| `VAD_SILENCE_MS` | `600` | Silence before segment boundary |
| `MIN_SPEECH_MS` | `200` | Minimum speech to process |
| `MAX_SEGMENT_SECONDS` | `30.0` | Force-split long segments |
| `CALIBRATION_DURATION_MS` | `2000` | Noise floor calibration period |

### Optional Features

| Env Var | Default | Purpose |
|---------|---------|---------|
| `TTS_ENABLED` | `false` | Enable Piper text-to-speech |
| `TTS_VOICE_ES` | `es_ES-carlfm-x_low` | Spanish voice model |
| `TTS_VOICE_EN` | `en_US-lessac-medium` | English voice model |
| `RECORDING_ENABLED` | `false` | Enable audio recording |
| `WS_PING_INTERVAL` | `30.0` | WebSocket ping interval (seconds) |
| `WS_MISSED_PINGS` | `3` | Missed pings before connection close |

### Infrastructure

| Env Var | Default | Purpose |
|---------|---------|---------|
| `DB_PATH` | `data/habla.db` | SQLite database path |
| `DATA_DIR` | `data` | Data directory for all runtime files |
| `LOG_LEVEL` | `INFO` | Console log level |

## HTTPS

Microphone access (`getUserMedia`) requires a secure context. Two options documented in `.dev/docs/HTTPS-Setup.md`:

1. **Tailscale HTTPS** (recommended): `tailscale cert` generates certificates for `{hostname}.tail1234.ts.net`. uvicorn serves HTTPS directly.
2. **Caddy reverse proxy**: Caddy handles TLS termination, proxies to uvicorn on localhost:8002.

## Design Rationale

**Why Tailscale?** The server is on a home network behind NAT. Tailscale provides a private overlay network with automatic TLS certificates, avoiding port forwarding and dynamic DNS.

**Why not a cloud GPU?** Cost and latency. A cloud GPU instance (A10G, T4) costs $0.50-1.50/hour. The MSI laptop is already paid for and has adequate GPU. Latency over Tailscale to a local machine is ~5-20ms vs 50-200ms to a cloud region.

**Why conservative VRAM?** The RTX 3060 in the target laptop has thermal throttling issues. Running at 42% VRAM capacity leaves margin for CUDA fragmentation, OS compositor, and thermal headroom. Running at 80%+ VRAM would risk OOM crashes during extended classroom sessions.

**Why all models always loaded?** The alternative — loading/unloading models per request — adds 5-15 seconds of latency per ASR call and risks GPU memory fragmentation. Always-loaded gives instant response at the cost of peak VRAM.

**Why Docker pinned Ollama to 0.5?** Ollama had breaking API changes between major versions. Pinning ensures the REST API contract (`/api/generate`, `/api/tags`) remains stable.

## Planned Direction

- **Cloud deployment evaluation** (Phase 10): Feasibility study for hosting as a multi-tenant service. Requires Phase 8 multi-client work first.
- **CPU-only Docker variant** (Phase 10.0): `python:3.11-slim` base without CUDA for cloud deployments using cloud ASR + cloud LLM.
- **Model cache volumes** (Phase 10.0): Persistent volumes for `~/.cache/huggingface` and `~/.cache/torch/hub` to avoid multi-GB model downloads on container restart.
