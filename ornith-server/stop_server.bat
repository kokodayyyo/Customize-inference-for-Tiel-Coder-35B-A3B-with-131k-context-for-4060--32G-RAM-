@echo off
REM ============================================================
REM  Stop the gateway and every llama-server process, free VRAM.
REM  Pure ASCII on purpose -- see start_server.bat for why.
REM
REM  Two things get stopped:
REM    1) the Python gateway listening on the API port (default 8000)
REM    2) all llama-server.exe processes (the model, holds the VRAM)
REM
REM  Note: /admin/stop in the web console only stops the MODEL and
REM  keeps the gateway alive, so you can start another model without
REM  reopening the console. This script stops everything.
REM
REM  Usage: stop_server.bat [api_port]
REM ============================================================
setlocal
chcp 65001 >nul

set "API_PORT=8000"
if not "%~1"=="" set "API_PORT=%~1"

echo [1/2] Stopping gateway on port %API_PORT% ...
set "FOUND="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /r /c:":%API_PORT% .*LISTENING"') do (
  if not "%%a"=="0" (
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
      echo       killed gateway pid %%a
      set "FOUND=1"
    )
  )
)
if not defined FOUND echo       no gateway listening on %API_PORT%

echo [2/2] Stopping llama-server ...
tasklist /FI "IMAGENAME eq llama-server.exe" 2>nul | find /I "llama-server.exe" >nul
if errorlevel 1 (
  echo       no llama-server process is running
) else (
  taskkill /F /IM llama-server.exe >nul 2>&1
  echo       llama-server stopped
)

echo.
echo Current GPU memory usage:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo.
echo Done. Run start_server.bat to bring it back.
endlocal
