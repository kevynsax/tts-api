# syntax=docker/dockerfile:1
# App image: just the code on top of the prebuilt deps base (Dockerfile.base).
# No pip install here, so this builds in seconds — ideal for per-commit CI.
#
# Build (after pushing the base image):
#   container build --platform linux/amd64 \
#     --build-arg BASE_IMAGE=registry.kevyn.com.br/ai-features/tts-base:1.0.0 \
#     -t registry.kevyn.com.br/ai-features/tts-server:1.0.0 -f Dockerfile.app .
ARG BASE_IMAGE=registry.kevyn.com.br/ai-features/tts-base:1.0.0
FROM ${BASE_IMAGE}

WORKDIR /app

COPY voices ./voices
COPY server_cpu.py ./server.py

ENV TTS_HOST=0.0.0.0 \
    TTS_PORT=8000 \
    TTS_MODEL=chatterbox \
    TTS_DEVICE=auto \
    HF_HOME=/data/huggingface

VOLUME ["/data/huggingface"]
EXPOSE 8000

CMD ["python", "server.py"]
