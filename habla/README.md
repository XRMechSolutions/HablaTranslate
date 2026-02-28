# Habla — Real-Time Bidirectional Speech Translation

Self-hosted Spanish ⇄ English speech translator with speaker diarization,
idiom detection, and vocabulary capture for language learning.

Runs on a single RTX 3060 (12GB) with ~5GB VRAM usage.

## Initial Setup (New Device)

A setup script automates most of this. From the project root, run as **Administrator**:

```
setup.bat
```

It checks prerequisites, installs what it can (ffmpeg, pip packages, PyTorch CUDA), downloads ML models, generates Tailscale HTTPS certs, and updates the startup scripts. Review its output for anything that needs manual attention.

The sections below cover what the script does and what still requires manual steps.

### Prerequisites

| Requirement | How to install | Verify |
|-------------|---------------|--------|
| NVIDIA GPU (12GB+ VRAM) | [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) + [CUDA Toolkit 12.4+](https://developer.nvidia.com/cuda-downloads) | `nvidia-smi` |
| Python 3.11+ | [python.org](https://www.python.org/downloads/) or Microsoft Store. Check "Add to PATH". | `python --version` |
| ffmpeg | `winget install ffmpeg` (restart terminal after) | `ffmpeg -version` |
| Tailscale | [tailscale.com/download](https://tailscale.com/download) | `tailscale status` |
| LM Studio **or** Ollama | [lmstudio.ai](https://lmstudio.ai) / [ollama.com](https://ollama.com) | See below |

### Manual Steps (cannot be automated)

**1. HuggingFace token + model terms**

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough), then accept the terms on both of these pages:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Add the token to `habla/.env`:
```
HF_TOKEN=hf_your_real_token_here
```

**2. LLM provider setup**

*Option A — LM Studio (default):*
- Install LM Studio and download your translation models (GGUF format)
- Models are stored in `C:\Users\<username>\.cache\lm-studio\models\`
- Update `habla/.env` with the paths on this machine:
  ```
  LMSTUDIO_EXECUTABLE=C:/Program Files/LM Studio/LM Studio.exe
  LMSTUDIO_MODEL=towerinstruct-mistral-7b-v0.2
  LMSTUDIO_MODEL_PATHS=C:/Users/<username>/.cache/lm-studio/models/.../model.gguf
  ```
- Start LM Studio and load your model before starting Habla

*Option B — Ollama:*
- Install Ollama, then: `ollama pull qwen3:4b`
- Set in `habla/.env`:
  ```
  LLM_PROVIDER=ollama
  ```

**3. PyTorch with CUDA**

The setup script attempts this, but if CUDA isn't detected, install manually:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Verify: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

### Quick Start (after setup)

```
start-habla.bat
```

Wait for **"Application startup complete"** in the console before opening the browser. First launch downloads/loads ML models and may take 1-2 minutes.

Access at: `https://<your-tailscale-hostname>:8002`

### Remote Access (Phone)

The server is accessed via [Tailscale](https://tailscale.com) VPN. Install Tailscale on your phone, sign in with the same account, and open the HTTPS URL. HTTPS is required for the browser to allow microphone access.

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
