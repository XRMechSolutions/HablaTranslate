@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Habla Translate - New Device Setup
echo ============================================================
echo.
echo This script checks prerequisites, installs what it can,
echo downloads ML models, and guides you through manual steps.
echo.
echo Run as Administrator for best results (Tailscale certs,
echo ffmpeg install, firewall rules).
echo.

set HABLA_DIR=%~dp0habla
set ERRORS=0
set WARNINGS=0

:: ============================================================
:: 1. Python
:: ============================================================
echo [1/10] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Python not found. Install Python 3.11+ from python.org or Microsoft Store.
    echo          Make sure "Add to PATH" is checked during install.
    set /a ERRORS+=1
    echo.
    echo   Cannot continue without Python. Install it and re-run this script.
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo   [OK] Python %%v
)
echo.

:: ============================================================
:: 2. CUDA / GPU
:: ============================================================
echo [2/10] Checking NVIDIA GPU and CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] nvidia-smi not found. Install NVIDIA drivers + CUDA Toolkit.
    echo          Download: https://developer.nvidia.com/cuda-downloads
    set /a ERRORS+=1
) else (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>nul
    echo   [OK] NVIDIA GPU detected
)
echo.

:: ============================================================
:: 3. ffmpeg
:: ============================================================
echo [3/10] Checking ffmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo   [MISS] ffmpeg not found. Attempting install via winget...
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo   [FAIL] winget install failed. Download manually from https://ffmpeg.org/download.html
        echo          Extract and add the bin folder to your system PATH.
        set /a ERRORS+=1
    ) else (
        echo   [OK] ffmpeg installed successfully.
        echo   [WARN] Restart your terminal after setup for PATH to update.
        set /a WARNINGS+=1
    )
) else (
    for /f "tokens=3" %%v in ('ffmpeg -version 2^>^&1 ^| findstr /r "^ffmpeg"') do echo   [OK] ffmpeg %%v
)
echo.

:: ============================================================
:: 4. Tailscale + HTTPS Certificates
:: ============================================================
echo [4/10] Checking Tailscale and HTTPS certificates...
tailscale status >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Tailscale not found or not running.
    echo          Install: https://tailscale.com/download
    set /a ERRORS+=1
    goto :skip_tailscale
)

:: Get FQDN via tailscale status --json
for /f "tokens=*" %%f in ('powershell -Command "(tailscale status --json | ConvertFrom-Json).Self.DNSName.TrimEnd('.')" 2^>nul') do set TS_FQDN=%%f

if not defined TS_FQDN (
    echo   [WARN] Could not detect Tailscale FQDN automatically.
    echo          Run: powershell -Command "(tailscale status --json | ConvertFrom-Json).Self.DNSName"
    set /a WARNINGS+=1
    goto :skip_tailscale
)

echo   [OK] Tailscale FQDN: %TS_FQDN%

:: Check if cert files already exist
set CERT_FILE=%~dp0%TS_FQDN%.crt
set KEY_FILE=%~dp0%TS_FQDN%.key

if exist "%CERT_FILE%" (
    echo   [OK] TLS certificate already exists
) else (
    echo   [INFO] Generating TLS certificate for %TS_FQDN%...
    tailscale cert --cert-file "%CERT_FILE%" --key-file "%KEY_FILE%" %TS_FQDN%
    if errorlevel 1 (
        echo   [FAIL] Certificate generation failed. Make sure you're running as Administrator.
        set /a ERRORS+=1
    ) else (
        echo   [OK] Certificate generated successfully
    )
)

:: Update start-habla.bat with correct hostname
echo   [INFO] Updating start-habla.bat with Tailscale hostname...
powershell -Command "(Get-Content '%~dp0start-habla.bat') -replace 'set CERT_FILE=%%~dp0.*\.crt', 'set CERT_FILE=%%~dp0%TS_FQDN%.crt' -replace 'set KEY_FILE=%%~dp0.*\.key', 'set KEY_FILE=%%~dp0%TS_FQDN%.key' -replace 'set TAILSCALE_HOST=.*', 'set TAILSCALE_HOST=%TS_FQDN%' | Set-Content '%~dp0start-habla.bat'"
echo   [OK] start-habla.bat updated

:skip_tailscale
echo.

:: ============================================================
:: 5. Windows Firewall
:: ============================================================
echo [5/10] Checking firewall rule for port 8002...
netsh advfirewall firewall show rule name="Habla Server" >nul 2>&1
if errorlevel 1 (
    echo   [INFO] Adding firewall rule for port 8002...
    netsh advfirewall firewall add rule name="Habla Server" dir=in action=allow protocol=TCP localport=8002 >nul 2>&1
    if errorlevel 1 (
        echo   [WARN] Could not add firewall rule. Run as Administrator, or add manually.
        set /a WARNINGS+=1
    ) else (
        echo   [OK] Firewall rule added
    )
) else (
    echo   [OK] Firewall rule exists
)
echo.

:: ============================================================
:: 6. Python dependencies
:: ============================================================
echo [6/10] Installing Python dependencies...
if exist "%HABLA_DIR%\requirements.txt" (
    pip install -r "%HABLA_DIR%\requirements.txt"
    if errorlevel 1 (
        echo   [WARN] Some packages failed. Check output above.
        set /a WARNINGS+=1
    ) else (
        echo   [OK] Python dependencies installed
    )
) else (
    echo   [FAIL] requirements.txt not found at %HABLA_DIR%\requirements.txt
    set /a ERRORS+=1
)
echo.

:: ============================================================
:: 7. PyTorch CUDA
:: ============================================================
echo [7/10] Checking PyTorch CUDA support...
python -c "import torch; cuda=torch.cuda.is_available(); print(f'  [OK] PyTorch {torch.__version__}, CUDA: {cuda}'); exit(0 if cuda else 1)" 2>nul
if errorlevel 1 (
    echo   [WARN] PyTorch missing or no CUDA support. Installing PyTorch with CUDA...
    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 (
        echo   [FAIL] PyTorch CUDA install failed.
        echo          Manual install: https://pytorch.org/get-started/locally/
        set /a ERRORS+=1
    ) else (
        python -c "import torch; print(f'  [OK] PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>nul
    )
)
echo.

:: ============================================================
:: 8. .env configuration
:: ============================================================
echo [8/10] Checking .env configuration...
if exist "%HABLA_DIR%\.env" (
    echo   [OK] .env file exists
    echo   [INFO] Verify these are correct for this machine:
    echo          %HABLA_DIR%\.env
    echo     - HF_TOKEN             (HuggingFace token)
    echo     - LMSTUDIO_EXECUTABLE  (LM Studio path)
    echo     - LMSTUDIO_MODEL_PATHS (GGUF model paths — machine-specific!)
) else (
    if exist "%HABLA_DIR%\.env.example" (
        copy "%HABLA_DIR%\.env.example" "%HABLA_DIR%\.env" >nul
        echo   [OK] Created .env from .env.example
        echo   [ACTION REQUIRED] Edit %HABLA_DIR%\.env and fill in:
        echo     - HF_TOKEN=hf_your_token_here
        echo     - LMSTUDIO_EXECUTABLE=C:/Program Files/LM Studio/LM Studio.exe
        echo     - LMSTUDIO_MODEL_PATHS=path/to/model.gguf
        set /a WARNINGS+=1
    ) else (
        echo   [FAIL] No .env or .env.example found
        set /a ERRORS+=1
    )
)
echo.

:: ============================================================
:: 9. Download ML Models
:: ============================================================
echo [9/10] Downloading ML models (this may take several minutes)...
echo.

:: --- 9a. Silero VAD (~2MB) ---
echo   [9a] Silero VAD model...
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True); print('  [OK] Silero VAD downloaded')" 2>nul
if errorlevel 1 (
    echo   [WARN] Silero VAD download failed. Will retry on first server start.
    set /a WARNINGS+=1
)
echo.

:: --- 9b. WhisperX / faster-whisper "small" model (~500MB) ---
echo   [9b] WhisperX small model (faster-whisper, ~500MB)...
python -c "from faster_whisper.utils import download_model; download_model('small'); print('  [OK] WhisperX small model downloaded')" 2>nul
if errorlevel 1 (
    :: Try alternative download method
    python -c "from huggingface_hub import snapshot_download; snapshot_download('guillaumekln/faster-whisper-small'); print('  [OK] WhisperX small model downloaded')" 2>nul
    if errorlevel 1 (
        echo   [WARN] WhisperX model download failed. Will download on first server start.
        echo          This is a ~500MB download and may cause the first startup to be slow.
        set /a WARNINGS+=1
    )
)
echo.

:: --- 9c. Pyannote diarization (~100MB, needs HF_TOKEN) ---
echo   [9c] Pyannote speaker diarization model (~100MB)...

:: Read HF_TOKEN from .env if it exists
set HF_TOKEN_FOUND=
if exist "%HABLA_DIR%\.env" (
    for /f "tokens=1,2 delims==" %%a in ('findstr /r "^HF_TOKEN=" "%HABLA_DIR%\.env"') do (
        set HF_TOKEN_VALUE=%%b
    )
)
if defined HF_TOKEN_VALUE (
    if "!HF_TOKEN_VALUE!"=="hf_your_token_here" (
        echo   [SKIP] HF_TOKEN is still the placeholder value.
        echo          Edit .env with your real token, then re-run setup.
        echo          Get a token at: https://huggingface.co/settings/tokens
        echo          Accept model terms at:
        echo            https://huggingface.co/pyannote/speaker-diarization-3.1
        echo            https://huggingface.co/pyannote/segmentation-3.0
        set /a WARNINGS+=1
    ) else (
        python -c "import os; os.environ['HF_TOKEN']='!HF_TOKEN_VALUE!'; from pyannote.audio import Pipeline; p=Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', token='!HF_TOKEN_VALUE!'); print('  [OK] Pyannote diarization model downloaded')" 2>nul
        if errorlevel 1 (
            echo   [WARN] Pyannote download failed. Common causes:
            echo          - HF_TOKEN is invalid
            echo          - Model terms not accepted at huggingface.co
            echo          Will retry on first server start.
            set /a WARNINGS+=1
        )
    )
) else (
    echo   [SKIP] No HF_TOKEN found in .env. Diarization won't work without it.
    echo          Get a token at: https://huggingface.co/settings/tokens
    set /a WARNINGS+=1
)
echo.

:: --- 9d. LM Studio models ---
echo   [9d] LM Studio translation models...
if exist "C:\Program Files\LM Studio\LM Studio.exe" (
    echo   [OK] LM Studio is installed
    where lms >nul 2>&1
    if errorlevel 1 (
        echo   [INFO] LM Studio CLI (lms) not in PATH. To download models from command line:
        echo          1. Open LM Studio
        echo          2. Search for and download your translation models
        echo          3. Update LMSTUDIO_MODEL_PATHS in .env with the GGUF file paths
        echo          Models are typically stored in: C:\Users\%USERNAME%\.cache\lm-studio\models\
    ) else (
        echo   [INFO] LM Studio CLI available. You can download models with:
        echo          lms get towerinstruct-mistral-7b-v0.2
        echo          Then update LMSTUDIO_MODEL_PATHS in .env
    )
) else (
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo   [WARN] No LLM provider found. Install one:
        echo          LM Studio: https://lmstudio.ai
        echo          Ollama:    https://ollama.com  (then: ollama pull qwen3:4b)
        set /a WARNINGS+=1
    ) else (
        echo   [OK] Ollama found. Checking for translation model...
        ollama list 2>nul | findstr "qwen3" >nul 2>&1
        if errorlevel 1 (
            echo   [INFO] Pulling qwen3:4b model for Ollama (~2.5GB)...
            ollama pull qwen3:4b
            if errorlevel 1 (
                echo   [WARN] Ollama model pull failed. Run manually: ollama pull qwen3:4b
                set /a WARNINGS+=1
            ) else (
                echo   [OK] qwen3:4b model downloaded
            )
        ) else (
            echo   [OK] Ollama qwen3 model available
        )
    )
)
echo.

:: ============================================================
:: 10. Data directories
:: ============================================================
echo [10/10] Creating data directories...
if not exist "%HABLA_DIR%\data" mkdir "%HABLA_DIR%\data"
if not exist "%HABLA_DIR%\data\audio" mkdir "%HABLA_DIR%\data\audio"
if not exist "%HABLA_DIR%\data\audio\recordings" mkdir "%HABLA_DIR%\data\audio\recordings"
echo   [OK] Data directories ready
echo.

:: ============================================================
:: Summary
:: ============================================================
echo ============================================================
echo   Setup Complete
echo ============================================================
echo.
if %ERRORS% GTR 0 (
    echo   %ERRORS% ERROR(s) - fix these before running the server
)
if %WARNINGS% GTR 0 (
    echo   %WARNINGS% WARNING(s) - review messages above
)
if %ERRORS% EQU 0 if %WARNINGS% EQU 0 (
    echo   Everything looks good!
)
echo.
echo   Checklist before first run:
echo     [ ] .env has real HF_TOKEN (not placeholder)
echo     [ ] .env has correct LMSTUDIO_MODEL_PATHS for this machine
echo     [ ] HuggingFace model terms accepted (links above)
echo     [ ] LM Studio running with models loaded (or Ollama running)
echo     [ ] Restart terminal if ffmpeg was just installed
echo.
echo   To start the server:
echo     start-habla.bat
echo.
echo   Wait for "Application startup complete" in the console
echo   before opening the browser.
echo.
pause
exit /b %ERRORS%
