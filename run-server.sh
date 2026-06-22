#!/usr/bin/env bash
# Start the OpenAI-compatible Chatterbox TTS server.
# Usage: ./run-server.sh            (defaults to 127.0.0.1:8000)
#        TTS_PORT=9000 ./run-server.sh
set -euo pipefail
cd "$(dirname "$0")"

# free the port if a previous run is still bound to it
PORT="${TTS_PORT:-8000}"
if lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT busy, killing previous server..."
  lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN | xargs -r kill -9 || true
  sleep 1
fi

exec ./.venv-mlx/bin/python server.py
