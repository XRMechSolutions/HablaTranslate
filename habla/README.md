# Habla — Real-Time Bidirectional Speech Translation

Self-hosted Spanish ⇄ English speech translator with speaker diarization,
idiom detection, and vocabulary capture for language learning.

Runs on a single RTX 3060 (12GB) with ~5GB VRAM usage.

## Quick Start

### Prerequisites

- NVIDIA GPU with 12GB+ VRAM
- [Ollama](https://ollama.com) installed
- Python 3.11+
- ffmpeg installed (`sudo apt install ffmpeg`)
- HuggingFace account + token (for Pyannote diarization)

### 1. Pull the LLM model

```bash
ollama pull qwen3:4b
```

### 2. Accept Pyannote model licenses

Visit these HuggingFace pages and accept the terms:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 3. Install and run

```bash
cd habla
pip install -r requirements.txt

# Set your HuggingFace token
export HF_TOKEN=hf_your_token_here

# Start the server
uvicorn server.main:app --host 0.0.0.0 --port 8002
```

### 4. Open on your phone

Open `http://YOUR_SERVER_IP:8002` in your phone's browser.
For remote access, set up WireGuard VPN.

## Docker Alternative

```bash
export HF_TOKEN=hf_your_token_here
docker compose up
```

## Usage

- **Push-to-talk**: Hold the mic button to record, release to translate
- **Text input**: Type directly to translate without speech
- **Toggle direction**: Tap `ES → EN` to switch to `EN → ES`
- **Toggle mode**: Tap `conversation` to switch to `classroom`
- **Name speakers**: Tap any speaker label to assign a name
- **Save vocab**: Tap ⭐ on idiom/correction cards to save for study
- **Review vocab**: Tap 📚 to open the vocabulary review page

## Architecture

- **WhisperX Small** (~1GB GPU) — always-loaded ASR
- **Pyannote 3.1** (CPU) — speaker diarization
- **Qwen3 4B Q3** (~2.5GB GPU) — contextual translation
- **Idiom pattern DB** (CPU) — fast regex-based idiom detection
- **Piper TTS** (CPU, optional) — speak translations aloud

Total GPU: ~5GB steady state. Safe for power-constrained hardware.

## Configuration

Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace access token |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `qwen3:4b` | Translation model |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Set to `cpu` to run Whisper on CPU |
