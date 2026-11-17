@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [Setup] Creating Python virtual environment...
  py -3 -m venv .venv || exit /b 1
  .venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1
  .venv\Scripts\python.exe -m playwright install chromium || exit /b 1
)
if not exist .env (
  copy .env.example .env >nul
  echo [INFO] Edit .env to set MOMA_API_KEY for real model calls. Demo mode works without a key.
)
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
