@echo off
setlocal

set PORT=8002
set HABLA_DIR=%~dp0habla
set CERT_FILE=%~dp0razorwhip.tailf87b45.ts.net.crt
set KEY_FILE=%~dp0razorwhip.tailf87b45.ts.net.key
set TAILSCALE_HOST=razorwhip.tailf87b45.ts.net
echo ============================================
echo  Habla Translator - Startup
echo ============================================
echo.

:: --- Check ffmpeg ---
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARN] ffmpeg not found in PATH. Audio recording will not work.
    echo.
)

:: --- Kill any existing server on our port ---
echo [INFO] Checking for existing server on port %PORT%...
powershell -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Write-Host '[INFO] Killing existing server (PID' $_.OwningProcess ')'; Stop-Process -Id $_.OwningProcess -Force }"

:: --- Start Habla server ---
echo [INFO] Starting Habla server on port %PORT%...
echo [INFO] Press Ctrl+C to stop.
echo.

:: --- Open browser after short delay ---
start "" cmd /c "timeout /t 5 /nobreak >nul & start https://%TAILSCALE_HOST%:%PORT%"

:: --- Set environment variables ---
set RECORDING_ENABLED=1
:: Secrets (HF_TOKEN, OPENAI_API_KEY, etc.) are loaded from habla\.env by Python at startup.

:: --- Launch uvicorn with HTTPS (blocks until Ctrl+C) ---
cd /d "%HABLA_DIR%"
if exist "%CERT_FILE%" (
    echo [INFO] Starting with HTTPS enabled and recording enabled
    python -m uvicorn server.main:app --host 0.0.0.0 --port %PORT% --ssl-keyfile "%KEY_FILE%" --ssl-certfile "%CERT_FILE%"
) else (
    echo [WARN] HTTPS certificates not found, starting without SSL
    echo [WARN] Microphone access will be blocked on remote devices
    python -m uvicorn server.main:app --host 0.0.0.0 --port %PORT%
)

echo.
echo [INFO] Habla server stopped.
pause
exit /b 0
