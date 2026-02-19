@echo off
REM Restart Habla server with all latest code changes
echo.
echo ============================================================
echo   Restarting Habla Server
echo ============================================================
echo.

cd /d "%~dp0"

echo Stopping any running instances...
taskkill /F /IM uvicorn.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting server with updated code...
echo Press Ctrl+C to stop the server
echo.

uvicorn server.main:app --host 0.0.0.0 --port 8002
