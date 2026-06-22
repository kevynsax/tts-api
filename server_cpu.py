#!/usr/bin/env python
"""
Containerizable OpenAI-compatible Chatterbox TTS server (PyTorch).

This mirrors server.py's HTTP API but uses the PyTorch `chatterbox-tts` package
instead of MLX, because MLX needs Apple Silicon + Metal and cannot run inside a
Linux container. It runs on CPU (or CUDA if available) — slower than the native
MLX server, but portable. The native server.py remains the fast path on the host.

Same routes as server.py:
    POST /v1/audio/speech   {model, input, voice, response_format, speed, language}
    GET  /v1/audio/voices
    GET  /health
    GET  /openapi.json, /docs   (FastAPI built-ins — used as the availability probe)

Env: CHATTERBOX_TORCH_DEVICE (auto|cpu|cuda), TTS_HOST (0.0.0.0), TTS_PORT (8000).
"""
from __future__ import annotations

import io
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

VOICES_DIR = Path(__file__).parent / "voices"
_lock = threading.Lock()
_state: dict = {}


def pick_device() -> str:
    pref = os.environ.get("CHATTERBOX_TORCH_DEVICE", "auto")
    if pref != "auto":
        return pref
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from chatterbox.tts import ChatterboxTTS
    device = pick_device()
    print(f"[server-cpu] loading Chatterbox on {device} ...")
    _state["model"] = ChatterboxTTS.from_pretrained(device=device)
    _state["sr"] = _state["model"].sr
    print("[server-cpu] model ready")
    yield
    _state.clear()


app = FastAPI(title="Chatterbox OpenAI-compatible TTS (torch)", version="1.0.0", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = 1.0
    language: str | None = None  # accepted for API parity (English model ignores it)
    exaggeration: float | None = None
    cfg_weight: float | None = None


_CONTENT_TYPES = {
    "wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg",
    "flac": "audio/flac", "opus": "audio/ogg", "aac": "audio/aac",
}
_FFMPEG_FMT = {"mp3": "mp3", "flac": "flac", "opus": "opus", "aac": "adts"}


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode_audio(samples: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    fmt = fmt.lower()
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported response_format '{fmt}'.")
    if fmt == "wav":
        return _wav_bytes(samples, sr), _CONTENT_TYPES["wav"]
    if fmt == "pcm":
        return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes(), _CONTENT_TYPES["pcm"]
    wav = _wav_bytes(samples, sr)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", _FFMPEG_FMT[fmt], "pipe:1"],
        input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg failed: {proc.stderr.decode()[:300]}")
    return proc.stdout, _CONTENT_TYPES[fmt]


def resolve_voice(voice: str) -> str | None:
    if not voice or voice in ("default", "chatterbox"):
        return None
    for ext in (".wav", ".mp3", ".flac", ".m4a"):
        p = VOICES_DIR / f"{voice}{ext}"
        if p.exists():
            return str(p)
    return None


def list_voices() -> list[str]:
    voices = ["default"]
    if VOICES_DIR.exists():
        voices += sorted({p.stem for p in VOICES_DIR.iterdir() if p.suffix in (".wav", ".mp3", ".flac", ".m4a")})
    return voices


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    model = _state.get("model")
    if model is None:
        raise HTTPException(503, "model not loaded yet")
    if not req.input.strip():
        raise HTTPException(400, "input is empty")

    kwargs: dict = {}
    ref = resolve_voice(req.voice)
    if ref:
        kwargs["audio_prompt_path"] = ref
    if req.exaggeration is not None:
        kwargs["exaggeration"] = req.exaggeration
    if req.cfg_weight is not None:
        kwargs["cfg_weight"] = req.cfg_weight

    with _lock:
        wav = model.generate(req.input, **kwargs)
    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
    sr = _state["sr"]
    duration = len(samples) / sr
    audio, content_type = encode_audio(samples, sr, req.response_format)
    return Response(
        content=audio,
        media_type=content_type,
        headers={
            "X-Audio-Duration-Seconds": f"{duration:.3f}",
            "Access-Control-Expose-Headers": "X-Audio-Duration-Seconds",
        },
    )


@app.get("/v1/audio/voices")
def voices():
    return {"voices": list_voices()}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "chatterbox", "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": "model" in _state}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("TTS_HOST", "0.0.0.0"), port=int(os.environ.get("TTS_PORT", "8000")))
