@echo off
setlocal

set PORT=8002
set HABLA_DIR=%~dp0habla

:: --- Load TAILSCALE_HOST from habla\.env ---
:: Set TAILSCALE_HOST in habla\.env (e.g., TAILSCALE_HOST=mylaptop.tail12345.ts.net)
:: Generate certs with: tailscale cert <your-fqdn>
set TAILSCALE_HOST=
if exist "%HABLA_DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%HABLA_DIR%\.env") do (
        if "%%A"=="TAILSCALE_HOST" set "TAILSCALE_HOST=%%B"
    )
)
if "%TAILSCALE_HOST%"=="" (
    echo [WARN] TAILSCALE_HOST not set in habla\.env — HTTPS will be disabled.
    echo [WARN] Add TAILSCALE_HOST=your-hostname.ts.net to habla\.env
)
set CERT_FILE=%~dp0%TAILSCALE_HOST%.crt
set KEY_FILE=%~dp0%TAILSCALE_HOST%.key
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

:: --- Disable NordLynx adapter if active (conflicts with Tailscale's 100.64.0.0/10 range) ---
set NORD_WAS_STOPPED=0
powershell -Command "$a = Get-NetAdapter -Name 'NordLynx' -ErrorAction SilentlyContinue; if ($a -and $a.Status -eq 'Up') { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [INFO] NordLynx adapter detected — disabling to prevent Tailscale conflict...
    echo [INFO] A UAC prompt may appear to grant admin access.
    powershell -Command "Start-Process powershell -ArgumentList '-Command','Get-Service *nord* -ErrorAction SilentlyContinue | Stop-Service -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2; Disable-NetAdapter -Name NordLynx -Confirm:$false -ErrorAction SilentlyContinue' -Verb RunAs -Wait"
    set NORD_WAS_STOPPED=1
    echo [INFO] NordLynx disabled. NordVPN will be restored when Habla stops.
) else (
    echo [INFO] NordLynx not active — no VPN conflict.
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

:: --- Restart NordVPN services if we stopped them ---
if "%NORD_WAS_STOPPED%"=="1" (
    echo [INFO] Restarting NordVPN services...
    powershell -Command "Start-Process powershell -ArgumentList '-Command','Enable-NetAdapter -Name NordLynx -Confirm:$false -ErrorAction SilentlyContinue; Get-Service *nord* -ErrorAction SilentlyContinue | Where-Object { $_.StartType -ne ''Disabled'' } | Start-Service -ErrorAction SilentlyContinue' -Verb RunAs -Wait"
    echo [INFO] NordVPN services restored.
)

pause
exit /b 0
