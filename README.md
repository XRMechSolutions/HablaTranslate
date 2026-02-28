# Habla — Real-Time Bidirectional Speech Translation

Self-hosted Spanish ⇄ English speech translator built for noisy classrooms, live conversations, and language learning — where Google Translate gives up.

## Why Habla?

Google Translate's live speech mode passes Spanish through untranslated, butchers idioms, gives up on complex sentences, and badly mishandles quiet speech and noisy backgrounds. Habla uses an LLM-in-the-middle architecture that sees conversation context and uses reasoning to fix ASR errors, handle idioms by meaning, and infer what was *meant* rather than transcribing what was *heard*.

## Features

- **Real-time bidirectional translation** — Spanish ⇄ English with live partial results during speech
- **Speaker diarization** — identifies and labels different speakers in conversation
- **Idiom detection** — dual-layer system: fast regex pattern DB + LLM catches novel phrases
- **Vocabulary capture** — save idioms, corrections, and interesting constructions for study
- **Classroom mode** — detects teacher corrections, flags vocab for language learners
- **Mobile-first PWA** — works on your phone over Tailscale VPN
- **Text-only fallback** — works without a microphone for typed translation

## Architecture

| Component | Model | Where | VRAM |
|-----------|-------|-------|------|
| ASR | WhisperX Small | GPU | ~1 GB |
| Diarization | Pyannote 3.1 | CPU | 0 |
| Translation | Qwen3 4B Q3_K_M (Ollama) | GPU | ~2.5 GB |
| VAD | Silero | CPU | 0 |
| Idiom DB | Regex patterns | CPU | 0 |
| **Total** | | | **~5 GB** |

Runs comfortably on a single RTX 3060 (12 GB) with ~7 GB headroom for stability.

```
Browser Audio (Opus) → WebSocket → ffmpeg → Silero VAD
  → WhisperX ASR → Pyannote Diarization → Ollama LLM → WebSocket → Browser
```

## Quick Start

### Prerequisites

- **NVIDIA GPU** with 12+ GB VRAM and recent drivers
- **Python 3.11+**
- **CUDA Toolkit 12.4+**
- **ffmpeg** — `winget install ffmpeg`
- **Ollama** — [ollama.com](https://ollama.com) — then `ollama pull qwen3:4b`
- **HuggingFace token** — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free, needed for Pyannote diarization)

### Setup

Run the setup script as Administrator (checks prerequisites, installs dependencies, downloads models):

```
setup.bat
```

See [SETUP.md](SETUP.md) for the full walkthrough and manual steps.

### Run

```
start-habla.bat
```

Open `https://<your-tailscale-hostname>:8002` on your phone or PC.

### Docker Alternative

```bash
export HF_TOKEN=hf_your_token_here
cd habla && docker compose up
```

## Usage

- **Push-to-talk** — hold the mic button to record, release to translate
- **Text input** — type directly to translate without speech
- **Toggle direction** — tap the direction button to switch ES ⇄ EN
- **Classroom mode** — detects corrections, flags vocab automatically
- **Save vocab** — tap the star on idiom/correction cards for spaced repetition study

## Documentation

- [SETUP.md](SETUP.md) — detailed installation walkthrough
- [ARCHITECTURE.md](ARCHITECTURE.md) — file tree, module dependencies, technology table

## Hardware

Designed for an RTX 3060 (12 GB) with known thermal constraints. The ~5 GB VRAM budget at ~42% capacity is intentional — large headroom prevents crashes on power-constrained hardware. Do not increase model sizes without careful budgeting.

## Tech Stack

Python 3.11 / FastAPI / WebSocket / WhisperX / Pyannote / Ollama / Silero VAD / SQLite (async, WAL, FTS5) / vanilla JS PWA

## License

[MIT](LICENSE)

## Support

If you find Habla useful, consider supporting development:

<!-- TODO: Add GitHub Sponsors or Buy Me a Coffee link -->
