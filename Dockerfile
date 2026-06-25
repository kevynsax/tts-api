# syntax=docker/dockerfile:1
# Containerized TTS Server (PyTorch) for Linux / x86_64 / Kubernetes.
#
# The native server.py uses MLX (Apple Silicon + Metal) and cannot run in a Linux
# container, so this image runs the PyTorch path (server_cpu.py). It serves the
# Chatterbox, Kokoro, Orpheus, and Higgs Audio v2 backends and runs on CPU or CUDA
# (auto-detected; Orpheus/Higgs are 3B LLMs and want a GPU; Higgs defaults to 4-bit
# bitsandbytes quantization, ~5 GB VRAM). OpenAudio/Fish is MLX-host-only and is not
# included here.
#
# Build (on an x86_64 builder, or with buildx for cross-build):
#   docker build --platform linux/amd64 -t <registry>/tts-server:latest .
#   docker push <registry>/tts-server:latest
#
# Run locally:
#   docker run -d --name tts-server -p 8000:8000 \
#     -v tts-cache:/data/huggingface --memory 8g --cpus 4 <registry>/tts-server:latest
#   (>=6-8GB RAM to load the torch model; first start downloads weights into the cache.)
#
# Build caching: the `--mount=type=cache` below keeps pip's wheel cache across
# builds (no re-download even when the layer is rebuilt). On a persistent builder
# the whole pip layer is also reused via the normal layer cache. For ephemeral CI
# runners, build the deps once into a base image instead — see Dockerfile.base.
FROM python:3.11-slim

# ffmpeg: compressed audio output. espeak-ng: Kokoro G2P fallback / non-English.
# git: some HF model repos need it.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git espeak-ng

WORKDIR /app

# Install torch first so chatterbox-tts/kokoro reuse it. Defaults to the CUDA
# (cu124) build for GPU nodes — fast inference. For a smaller CPU-only image,
# build with: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX} torch==2.6.0 torchaudio==2.6.0

# If pip can't co-resolve chatterbox-tts + kokoro (they pin different dep
# versions), drop the `kokoro "misaki[en]"` line: the server still runs and the
# Chatterbox backend works; only the Kokoro backend reports an error on load.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        chatterbox-tts \
        kokoro "misaki[en]" \
        snac \
        bitsandbytes accelerate \
        soundfile fastapi "uvicorn[standard]" setproctitle

# Higgs Audio v2 is native in transformers >=5.3 (docs target 5.12); upgrade after
# the others install so a conflicting pin from chatterbox-tts/kokoro is visible here.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U "transformers>=5.3.0"

# Pre-download the spaCy English model misaki uses (avoids a first-request stall).
RUN python -m spacy download en_core_web_sm || true

COPY voices ./voices
COPY server_cpu.py ./server.py

ENV TTS_HOST=0.0.0.0 \
    TTS_PORT=8000 \
    TTS_MODEL=chatterbox \
    TTS_DEVICE=auto \
    HF_HOME=/data/huggingface

# Persist model weights here (mount a volume / PVC to avoid re-downloading).
VOLUME ["/data/huggingface"]
EXPOSE 8000

CMD ["python", "server.py"]
