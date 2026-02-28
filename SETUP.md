# Habla Translate — New Device Setup Guide

Complete checklist for getting Habla running on a new Windows machine.

**Quick start:** Run `setup.bat` as Administrator — it checks/installs most prerequisites automatically and tells you what's missing.

---

## Prerequisites

### 1. Python 3.11+

Install from [python.org](https://www.python.org/downloads/) or the Microsoft Store. Ensure "Add to PATH" is checked.

### 2. NVIDIA Drivers + CUDA Toolkit

WhisperX runs on GPU. Install:
- [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) for your GPU
- [CUDA Toolkit 12.4+](https://developer.nvidia.com/cuda-downloads)

Verify: `nvidia-smi` should show your GPU.

### 3. PyTorch with CUDA

The `requirements.txt` installs PyTorch, but it may default to CPU-only. Install the CUDA version explicitly:

```
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify:
```
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

### 4. ffmpeg

Required for audio decoding. Install via:
```
winget install ffmpeg
```
Restart your terminal after installing so it's in PATH.

### 5. Tailscale (for HTTPS + remote access)

Install from [tailscale.com/download](https://tailscale.com/download). Sign in with your account.

### 6. LLM Provider (pick one)

**LM Studio** (default):
- Install from [lmstudio.ai](https://lmstudio.ai)
- **Known compatible versions:** Habla uses the OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`). Any LM Studio version with this API works. Tested with LM Studio 0.3.x. Versions 0.2.x and earlier may have different API behavior.
- Download your translation models (GGUF format). Recommended:
  - `TowerInstruct-Mistral-7B-v0.2` (Q3_K_M) — primary translation model
  - A smaller model for quick/partial translations (e.g., `Ganymede-Llama-3.3-3B-Preview`)
- Models are stored in `C:\Users\<username>\.cache\lm-studio\models\`
- Note the full GGUF file paths — you'll need them for `.env`

**Ollama** (alternative):
- Install from [ollama.com](https://ollama.com)
- Pull model: `ollama pull qwen3:4b`
- Set `LLM_PROVIDER=ollama` in `habla/.env`

---

## Setup Steps

### Step 1: Clone the repo

```
git clone <your-repo-url> C:\Projects\HablaTranslate
cd C:\Projects\HablaTranslate
```

### Step 2: Install Python dependencies

```
cd habla
pip install -r requirements.txt
```

If PyTorch CUDA wasn't picked up, install it separately (see prerequisite 3 above).

### Step 3: Configure `.env`

```
copy habla\.env.example habla\.env
```

Edit `habla\.env` and fill in:

| Variable | Required | Notes |
|----------|----------|-------|
| `HF_TOKEN` | Yes (for diarization) | Get at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept terms for [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). |
| `LMSTUDIO_EXECUTABLE` | If using LM Studio | Full path, e.g. `C:/Program Files/LM Studio/LM Studio.exe` |
| `LMSTUDIO_MODEL_PATHS` | If using LM Studio | Semicolon-separated GGUF paths. These are machine-specific! |
| `LMSTUDIO_MODEL` | If using LM Studio | Model identifier |
| `OPENAI_API_KEY` | If using OpenAI | Your OpenAI API key |

### Step 4: Set up HTTPS via Tailscale

HTTPS is required for microphone access from remote devices (browsers block `getUserMedia` on non-HTTPS origins).

1. Find your Tailscale machine name:
   ```
   tailscale status
   ```
   Note the hostname (e.g., `mylaptop`).

2. Get your full domain:
   ```
   powershell -Command "(tailscale status --json | ConvertFrom-Json).Self.DNSName.TrimEnd('.')"
   ```
   Example output: `mylaptop.tail12345.ts.net`

3. Generate certificates (run as **Administrator**):
   ```
   tailscale cert --cert-file "C:\Projects\HablaTranslate\<FQDN>.crt" --key-file "C:\Projects\HablaTranslate\<FQDN>.key" <FQDN>
   ```
   Replace `<FQDN>` with your full domain from step 2.

4. Update `start-habla.bat` — edit these three lines near the top:
   ```
   set CERT_FILE=%~dp0<FQDN>.crt
   set KEY_FILE=%~dp0<FQDN>.key
   set TAILSCALE_HOST=<FQDN>
   ```

   Or just run `setup.bat` — it does steps 2-4 automatically.

### Step 5: Verify and launch

```
start-habla.bat
```

Wait for the log to show **"Application startup complete"** before opening the browser. The first launch loads ML models and may take 30-60 seconds.

Access at: `https://<your-tailscale-fqdn>:8002`

---

## Troubleshooting

### Browser says "can't reach this page" / ERR_CONNECTION_CLOSED
- **Server not ready yet.** Wait for "Application startup complete" in the console, then refresh.
- **Firewall blocking port 8002.** Run as Administrator:
  ```
  netsh advfirewall firewall add rule name="Habla Server" dir=in action=allow protocol=TCP localport=8002
  ```

### WhisperX or Pyannote fails to load (weights_only error)
PyTorch 2.6+ changed `torch.load` defaults, breaking older versions of WhisperX and Pyannote. Fix by upgrading to the compatible stack:
```
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install whisperx==3.8.1
```
This installs PyTorch 2.8.0, WhisperX 3.8.1, and pyannote-audio 4.0.4 — all compatible with the new `weights_only=True` default.

### No microphone access on phone
The browser requires HTTPS for `getUserMedia`. Verify:
- You're accessing via `https://` (not `http://`)
- The Tailscale cert matches the hostname in the URL
- Tailscale is connected on both the server and the phone

### LM Studio models not found
The `LMSTUDIO_MODEL_PATHS` in `.env` are full filesystem paths and are **machine-specific**. Update them to point to the GGUF files on the new machine. Check: `C:\Users\<username>\.cache\lm-studio\models\`

### Tailscale connection times out (NordVPN / WireGuard-based VPN conflict)
If you can reach the server via `https://localhost:8002` but **not** via the Tailscale hostname, and especially if `ping 100.x.x.x` (your own Tailscale IP) times out, a VPN adapter is likely hijacking Tailscale's IP range.

**Cause:** NordVPN's NordLynx adapter (and similar WireGuard-based VPNs) assigns itself an IP in the `100.64.0.0/10` CGNAT range — the same range Tailscale uses. The adapter stays active and claims routes **even when the VPN is "disconnected"**, stealing traffic meant for Tailscale.

**Diagnose:**
```
ipconfig | findstr /C:"NordLynx" /C:"100.101"
route print 100.*
```
If you see NordLynx with a `100.x.x.x` address and a `100.64.0.0 / 255.192.0.0` route pointing at it, that's the problem.

**Fix (run PowerShell as Administrator):**
```powershell
# Stop NordVPN services and disable the adapter
Get-Service *nord* | Stop-Service -Force -ErrorAction SilentlyContinue
Disable-NetAdapter -Name "NordLynx" -Confirm:$false
# Also disable other Nord adapters if present
Disable-NetAdapter -Name "OpenVPN Data Channel Offload for NordVPN" -Confirm:$false -ErrorAction SilentlyContinue
```

NordVPN will re-enable the adapter when you next connect to it. `start-habla.bat` handles this automatically — it disables NordLynx on startup (with a UAC prompt) and restores it when the server stops. If you need to fix it manually, use the PowerShell commands above.

Note: NordVPN's split tunneling is app-based only — it cannot exclude IP ranges, so there's no way to exempt Tailscale traffic. Switching NordVPN's protocol to OpenVPN also doesn't help — NordLynx stays active as a system tunnel regardless of the selected protocol.

### Tailscale cert generation fails
- Must run as Administrator
- Tailscale must be connected (not just installed)
- MagicDNS must be enabled in your Tailscale admin console

---

## Files That Are Machine-Specific (not in git)

| File | Purpose | How to create |
|------|---------|---------------|
| `habla/.env` | Secrets, model paths, provider config | Copy from `habla/.env.example` |
| `*.crt`, `*.key` | TLS certificates from Tailscale | `tailscale cert` command |
| `habla/data/habla.db` | SQLite database | Created automatically on first run |
| `habla/data/audio/recordings/` | Recorded audio clips | Created automatically |
