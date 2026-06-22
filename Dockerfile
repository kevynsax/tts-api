# Containerized Chatterbox TTS (PyTorch / CPU).
#
# NOTE: the native server.py uses MLX, which requires Apple Silicon + Metal and
# cannot run in a Linux container. This image therefore runs the PyTorch path
# (server_cpu.py) — portable but CPU-bound (slower). For full speed, run the
# native MLX server on the host (./run-server.sh) and reverse-proxy to it.
#
# Build:  container build -t chatterbox-tts .
# Run:    container run -d --name chatterbox-tts --memory 6g --cpus 4 chatterbox-tts
#         (needs >=6GB RAM to load the torch model; the 1GB default OOMs.
#          First start downloads the weights (~2GB) into the HF cache.)
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# chatterbox-tts pulls torch/torchaudio/transformers; soundfile for wav I/O.
RUN pip install --no-cache-dir chatterbox-tts soundfile fastapi "uvicorn[standard]"

# Clone reference voices (used for voice cloning by name).
COPY voices ./voices
COPY server_cpu.py ./server.py

ENV TTS_HOST=0.0.0.0 \
    TTS_PORT=8000 \
    HF_HOME=/app/.cache/huggingface

EXPOSE 8000

# First start downloads the model weights (~2GB) into the HF cache.
CMD ["python", "server.py"]
