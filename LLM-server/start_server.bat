@echo off
REM ============================================================
REM  Local inference service  --  console-only launcher
REM
REM  This starts ONLY the gateway + web console. No model is
REM  loaded, so it comes up in about a second. Pick a model in
REM  the console and it will be loaded then.
REM
REM      console : http://127.0.0.1:8000/ui
REM
REM  Passing --model/--ngl/... here does NOT load anything; to
REM  preload a model as well, use:  main.py serve
REM
REM  IMPORTANT: keep this file pure ASCII.
REM  cmd.exe parses .bat files using the system ANSI codepage (GBK
REM  on Chinese Windows). Non-ASCII bytes here get mis-decoded and
REM  break the script. All Chinese messages are printed by main.py.
REM
REM  Usage    : start_server.bat [--api-key sk-xxx] [--port 8000]
REM  Stop     : stop_server.bat   (stops console + any model)
REM  Health   : doctor.bat        (GPU / runtime / VRAM budget)
REM  Runtime  : bundled in runtime\llama.cpp\backends\  (no LM Studio needed)
REM ============================================================
setlocal
REM Switch console to UTF-8 FIRST so the Chinese banner from main.py renders.
chcp 65001 >nul
cd /d "%~dp0"

REM ---- Python interpreter (this service does NOT need torch) ----
if "%LLM_PYTHON%"=="" (
  set "LLM_PYTHON=D:\anaconda\envs\test1\python.exe"
)
if not exist "%LLM_PYTHON%" (
  echo [ERROR] Python not found: %LLM_PYTHON%
  echo         Set the LLM_PYTHON environment variable to a valid python.exe
  pause
  exit /b 1
)

echo.
echo   Starting console (no model loaded) ...
echo.

REM --no-autostart is the whole point: gateway + console only.
"%LLM_PYTHON%" main.py serve --no-autostart %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [ERROR] Service exited with code %RC%
  pause
)
endlocal
exit /b %RC%
