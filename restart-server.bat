@echo off
echo Stopping existing server on port 8002...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8002"') do (
    echo Killing PID %%a
    taskkill /F /PID %%a
)
timeout /t 2 /nobreak >nul
echo Starting server with HTTPS and recording...
call start-habla.bat
