@echo off
REM ============================================================
REM  Environment self-check: GPU, llama.cpp runtime, MoE placement
REM  budget, config sanity. Safe to run anytime -- does not load
REM  the model. Pure ASCII on purpose (see start_server.bat).
REM ============================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if "%LLM_PYTHON%"=="" (
  set "LLM_PYTHON=D:\anaconda\envs\test1\python.exe"
)
if not exist "%LLM_PYTHON%" (
  echo [ERROR] Python not found: %LLM_PYTHON%
  echo         Set the LLM_PYTHON environment variable to a valid python.exe
  pause
  exit /b 1
)

"%LLM_PYTHON%" main.py doctor
echo.
pause
endlocal
