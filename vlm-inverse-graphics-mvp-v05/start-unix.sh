#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m playwright install chromium
fi
if [ ! -f .env ]; then cp .env.example .env; fi
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
