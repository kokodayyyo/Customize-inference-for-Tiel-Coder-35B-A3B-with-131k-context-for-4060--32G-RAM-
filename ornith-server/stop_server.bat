@echo off
REM ============================================================
REM  Stop every llama-server process and free VRAM.
REM  Pure ASCII on purpose -- see start_server.bat for why.
REM ============================================================
setlocal
chcp 65001 >nul

echo Checking for running llama-server processes ...
tasklist /FI "IMAGENAME eq llama-server.exe" 2>nul | find /I "llama-server.exe" >nul
if errorlevel 1 (
  echo No llama-server process is running.
) else (
  taskkill /F /IM llama-server.exe
  echo llama-server stopped.
)
echo.
echo Current GPU memory usage:
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
endlocal
