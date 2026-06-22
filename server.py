#!/usr/bin/env python
"""
OpenAI-compatible TTS server backed by Chatterbox (MLX, Apple Silicon).

Exposes the same shape as OpenAI's audio API so existing OpenAI clients work
by just pointing base_url at this server:

    POST /v1/audio/speech      {model, input, voice, response_format, speed}
    GET  /v1/models
    GET  /v1/audio/voices      (extra: list local clone voices)
    GET  /health

Run:
    source .venv-mlx/bin/activate
    python server.py                 # serves on http://127.0.0.1:8000

Voices:
    Drop a reference clip at voices/<name>.wav (or .mp3) and request
    "voice": "<name>" to clone it. Any unknown voice falls back to the
    built-in default Chatterbox voice.

Env:
    CHATTERBOX_MODEL   default "mlx-community/chatterbox-fp16"
    TTS_HOST           default "127.0.0.1"
    TTS_PORT           default "8000"
    TTS_API_KEY        if set, require "Authorization: Bearer <key>"
"""
from __future__ import annotations

import asyncio
import io
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL_ID = os.environ.get("CHATTERBOX_MODEL", "mlx-community/chatterbox-fp16")
VOICES_DIR = Path(__file__).parent / "voices"
API_KEY = os.environ.get("TTS_API_KEY")

# MLX GPU streams are thread-local. Pin ALL MLX work (model load + every
# generation) to a single dedicated worker thread so the stream is consistent.
# max_workers=1 also serializes generation for free.
_mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_state: dict = {}


def _load_model_on_worker():
    from mlx_audio.tts.utils import load_model
    print(f"[server] loading {MODEL_ID} ...")
    _state["model"] = load_model(MODEL_ID)
    print("[server] model ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_mlx, _load_model_on_worker)
    yield
    _state.clear()
    _mlx.shutdown(wait=False)


app = FastAPI(title="Chatterbox OpenAI-compatible TTS", version="1.0.0", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = 1.0
    # ISO 639-1 code (en, pt, es, ...). Falsy/"unknown" => English.
    language: str | None = None
    # Chatterbox extras (optional, ignored by standard OpenAI clients)
    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None


# ---- audio helpers ---------------------------------------------------------

_CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}
# ffmpeg muxer name per format
_FFMPEG_FMT = {"mp3": "mp3", "flac": "flac", "opus": "opus", "aac": "adts"}


def _wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def encode_audio(samples: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    fmt = fmt.lower()
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported response_format '{fmt}'. "
                                 f"Use one of {sorted(_CONTENT_TYPES)}.")
    if fmt == "wav":
        return _wav_bytes(samples, sr), _CONTENT_TYPES["wav"]
    if fmt == "pcm":  # raw 16-bit LE mono @ sr
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return pcm, _CONTENT_TYPES["pcm"]
    # compressed formats via ffmpeg
    wav = _wav_bytes(samples, sr)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", _FFMPEG_FMT[fmt], "pipe:1"],
        input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg failed: {proc.stderr.decode()[:300]}")
    return proc.stdout, _CONTENT_TYPES[fmt]


def resolve_voice(voice: str) -> str | None:
    """Return a reference-audio path to clone, or None for the default voice."""
    if not voice or voice in ("default", "chatterbox"):
        return None
    for ext in (".wav", ".mp3", ".flac", ".m4a"):
        p = VOICES_DIR / f"{voice}{ext}"
        if p.exists():
            return str(p)
    return None  # unknown voice -> default


def synthesize(req: SpeechRequest) -> tuple[bytes, str, float]:
    model = _state.get("model")
    if model is None:
        raise HTTPException(503, "model not loaded yet")
    if not req.input.strip():
        raise HTTPException(400, "input is empty")

    ref_audio = resolve_voice(req.voice)
    lang = (req.language or "").strip().lower()
    kwargs: dict = {"speed": req.speed, "lang_code": lang if lang and lang != "unknown" else "en"}
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
    if req.exaggeration is not None:
        kwargs["exaggeration"] = req.exaggeration
    if req.cfg_weight is not None:
        kwargs["cfg_weight"] = req.cfg_weight
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature

    chunks: list[np.ndarray] = []
    sr = 24000
    for result in model.generate(text=req.input, verbose=False, **kwargs):
        sr = result.sample_rate or sr
        chunks.append(np.asarray(result.audio, dtype=np.float32))
    if not chunks:
        raise HTTPException(500, "no audio produced")
    samples = np.concatenate(chunks)
    duration = len(samples) / sr
    audio, content_type = encode_audio(samples, sr, req.response_format)
    return audio, content_type, duration


def _check_auth(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


# ---- routes ----------------------------------------------------------------

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, authorization: str | None = Header(default=None)):
    # Dispatch generation to the dedicated MLX worker thread (same thread that
    # loaded the model) so the GPU stream is valid. Awaiting keeps the event loop
    # free to answer /health etc. while audio renders.
    _check_auth(authorization)
    loop = asyncio.get_running_loop()
    audio, content_type, duration = await loop.run_in_executor(_mlx, synthesize, req)
    return Response(
        content=audio,
        media_type=content_type,
        headers={
            "X-Audio-Duration-Seconds": f"{duration:.3f}",
            "Access-Control-Expose-Headers": "X-Audio-Duration-Seconds",
        },
    )


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": "chatterbox", "object": "model", "owned_by": "local"},
        {"id": MODEL_ID, "object": "model", "owned_by": "local"},
    ]}


@app.get("/v1/audio/voices")
def list_voices():
    voices = ["default"]
    if VOICES_DIR.exists():
        voices += sorted({p.stem for p in VOICES_DIR.iterdir()
                          if p.suffix in (".wav", ".mp3", ".flac", ".m4a")})
    return {"voices": voices}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": "model" in _state}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TTS_HOST", "127.0.0.1")
    port = int(os.environ.get("TTS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
