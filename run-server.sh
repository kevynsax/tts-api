#!/usr/bin/env bash
# Start the OpenAI-compatible Chatterbox TTS server.
# Usage: ./run-server.sh            (defaults to 0.0.0.0:8000, reachable on the LAN)
#        TTS_HOST=127.0.0.1 ./run-server.sh   (localhost only)
#        TTS_PORT=9000 ./run-server.sh
set -euo pipefail
cd "$(dirname "$0")"

export TTS_HOST="${TTS_HOST:-0.0.0.0}"

# free the port if a previous run is still bound to it
PORT="${TTS_PORT:-8000}"
if lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT busy, killing previous server..."
  lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN | xargs -r kill -9 || true
  sleep 1
fi

exec ./.venv-mlx/bin/python server.py
