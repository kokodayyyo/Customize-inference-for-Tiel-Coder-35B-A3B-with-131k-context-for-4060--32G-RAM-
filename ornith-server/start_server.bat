@echo off
REM ============================================================
REM  Tile-35B-A3B  local inference API  --  one-click launcher
REM
REM  IMPORTANT: keep this file pure ASCII.
REM  cmd.exe parses .bat files using the system ANSI codepage (GBK
REM  on Chinese Windows). Non-ASCII bytes here get mis-decoded and
REM  break the script. All Chinese messages are printed by main.py.
REM
REM  Placement: MoE experts (14.12 GiB) in RAM,
REM             attention + 128K KV cache (1.33 GiB) in VRAM.
REM  Measured : decode ~28 tok/s, prefill ~1000 tok/s.
REM
REM  Usage    : start_server.bat [--api-key sk-xxx] [--port 8000]
REM  Health   : main.py doctor      (GPU / runtime / VRAM budget)
REM  Runtime  : bundled in runtime\llama.cpp\backends\  (no LM Studio needed)
REM ============================================================
setlocal
REM Switch console to UTF-8 FIRST so the Chinese banner from main.py renders.
chcp 65001 >nul
cd /d "%~dp0"

REM ---- Python interpreter (this service does NOT need torch) ----
if "%ORNITH_PYTHON%"=="" (
  set "ORNITH_PYTHON=D:\anaconda\envs\test1\python.exe"
)
if not exist "%ORNITH_PYTHON%" (
  echo [ERROR] Python not found: %ORNITH_PYTHON%
  echo         Set the ORNITH_PYTHON environment variable to a valid python.exe
  pause
  exit /b 1
)

echo.
echo   Loading model, please wait about 15 seconds ...
echo.

"%ORNITH_PYTHON%" main.py serve %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [ERROR] Service exited with code %RC%
  pause
)
endlocal
exit /b %RC%
