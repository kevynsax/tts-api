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
import ctypes
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path


def _detach_from_dock() -> None:
    """Run as a background process: no Dock tile, no window-server connection.

    When launched as a child of the menu-bar app, the framework Python otherwise
    registers with the window server (Metal init) and flashes a Dock tile on the
    main display. TransformProcessType -> background suppresses that.
    """
    if sys.platform != "darwin":
        return
    try:
        appservices = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")

        class ProcessSerialNumber(ctypes.Structure):
            _fields_ = [("highLongOfPSN", ctypes.c_uint32),
                        ("lowLongOfPSN", ctypes.c_uint32)]

        psn = ProcessSerialNumber(0, 2)  # kCurrentProcess
        appservices.TransformProcessType(ctypes.byref(psn), 2)  # kProcessTransformToBackgroundApplication
    except Exception:
        pass


_detach_from_dock()

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from setproctitle import setproctitle
    setproctitle("tts-server")
except ImportError:
    pass

# Switchable models. `key` is the short id clients pass; `repo` is the MLX repo.
MODEL_CATALOG = [
    {"key": "chatterbox", "label": "Chatterbox", "repo": "mlx-community/chatterbox-fp16"},
    {"key": "openaudio", "label": "OpenAudio (Fish S2 Pro)", "repo": "mlx-community/fish-audio-s2-pro-bf16"},
    {"key": "kokoro", "label": "Kokoro", "repo": "mlx-community/Kokoro-82M-bf16"},
    {"key": "orpheus", "label": "Orpheus", "repo": "mlx-community/orpheus-3b-0.1-ft-4bit"},
]
KNOWN_MODELS = {m["key"]: m["repo"] for m in MODEL_CATALOG}

MODEL_ID = os.environ.get("TTS_MODEL") or os.environ.get(
    "CHATTERBOX_MODEL", "mlx-community/chatterbox-fp16")
MODEL_ID = KNOWN_MODELS.get(MODEL_ID.lower(), MODEL_ID)


def _is_fish(model_id: str) -> bool:
    m = model_id.lower()
    return "fish" in m or "openaudio" in m


def _is_kokoro(model_id: str) -> bool:
    return "kokoro" in model_id.lower()


def _is_orpheus(model_id: str) -> bool:
    return "orpheus" in model_id.lower()


def _backend(model_id: str) -> str:
    if _is_kokoro(model_id):
        return "kokoro"
    if _is_fish(model_id):
        return "fish"
    if _is_orpheus(model_id):
        return "orpheus"
    return "chatterbox"


IS_FISH = _is_fish(MODEL_ID)
IS_KOKORO = _is_kokoro(MODEL_ID)
IS_ORPHEUS = _is_orpheus(MODEL_ID)
VOICES_DIR = Path(__file__).parent / "voices"
API_KEY = os.environ.get("TTS_API_KEY")

# MLX GPU streams are thread-local. Pin ALL MLX work (model load + every
# generation) to a single dedicated worker thread so the stream is consistent.
# max_workers=1 also serializes generation for free.
_mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_state: dict = {}

# Load lifecycle status, readable from any thread.
#   state: "loading" | "ready" | "error"
_status_lock = threading.Lock()
_status: dict = {"state": "loading", "model": MODEL_ID, "error": None}


def _set_status(**kw) -> None:
    with _status_lock:
        _status.update(kw)


def get_status() -> dict:
    with _status_lock:
        return dict(_status)


def _patch_kokoro_vocoder() -> None:
    """Work around an mlx-audio 0.4.4 bug: SineGen._f02sine loses a few frames to
    integer rounding when the sequence length isn't divisible by upsample_scale,
    so its output length != uv length and the vocoder crashes with a broadcast
    error for many inputs. Pad the sine waveform back to f0 length (what the
    divisible/working case produces naturally)."""
    try:
        from mlx_audio.tts.models.kokoro import istftnet as _ist
        import mlx.core as mx
    except Exception:
        return
    if getattr(_ist.SineGen, "_tts_patched", False):
        return

    def _call(self, f0):
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        target, cur = uv.shape[1], sine_waves.shape[1]
        if cur > target:
            sine_waves = sine_waves[:, :target, :]
        elif cur < target:
            pad = mx.broadcast_to(sine_waves[:, -1:, :],
                                  (sine_waves.shape[0], target - cur, sine_waves.shape[2]))
            sine_waves = mx.concatenate([sine_waves, pad], axis=1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

    _ist.SineGen.__call__ = _call
    _ist.SineGen._tts_patched = True
    print("[server] applied kokoro vocoder length-mismatch patch")


def _load_model_on_worker(model_id: str) -> None:
    global MODEL_ID, IS_FISH, IS_KOKORO, IS_ORPHEUS
    from mlx_audio.tts.utils import load_model
    backend = _backend(model_id)
    if backend == "kokoro":
        _patch_kokoro_vocoder()
    _set_status(state="loading", model=model_id, error=None)
    print(f"[server] loading {model_id} (backend={backend}) ...")
    try:
        model = load_model(model_id)
    except Exception as e:  # keep any previously loaded model usable
        msg = f"{type(e).__name__}: {e}"
        print(f"[server] model load FAILED for {model_id}: {msg}")
        if "model" in _state:
            _set_status(state="ready", model=MODEL_ID, error=msg)
        else:
            _set_status(state="error", model=model_id, error=msg)
        return
    _state["model"] = model
    MODEL_ID = model_id
    IS_FISH = _is_fish(model_id)
    IS_KOKORO = _is_kokoro(model_id)
    IS_ORPHEUS = _is_orpheus(model_id)
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    _set_status(state="ready", model=model_id, error=None)
    print(f"[server] ffmpeg: {FFMPEG or 'NOT FOUND'}")
    print(f"[server] model ready: {model_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    # Load in the background so the server answers /health (state="loading")
    # immediately instead of blocking startup until the model is ready.
    loop.run_in_executor(_mlx, _load_model_on_worker, MODEL_ID)
    yield
    _state.clear()
    _mlx.shutdown(wait=False)


app = FastAPI(title="TTS Server (OpenAI-compatible)", version="1.0.0", lifespan=lifespan)


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
    # Fish/OpenAudio voice cloning: transcript of the reference clip. If omitted,
    # falls back to voices/<voice>.txt when present.
    ref_text: str | None = None


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
# Explicit high bitrates so lossy codecs don't add their own shimmer on top of the
# vocoder. flac is lossless and takes none. opus is efficient, so 96k is transparent.
_FFMPEG_BITRATE = {"mp3": "256k", "aac": "256k", "opus": "96k"}


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if os.path.exists(p):
            return p
    return None


FFMPEG = _find_ffmpeg()


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
    if not FFMPEG:
        raise HTTPException(500, "ffmpeg not found; install it (brew install ffmpeg) "
                                 "or request response_format 'wav'/'pcm'.")
    wav = _wav_bytes(samples, sr)
    bitrate = ["-b:a", _FFMPEG_BITRATE[fmt]] if fmt in _FFMPEG_BITRATE else []
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-f", _FFMPEG_FMT[fmt], *bitrate, "pipe:1"],
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


def resolve_ref_text(voice: str) -> str | None:
    """Return the transcript for a cloned voice from voices/<voice>.txt, if any."""
    if not voice or voice in ("default", "chatterbox"):
        return None
    p = VOICES_DIR / f"{voice}.txt"
    if p.exists():
        text = p.read_text(encoding="utf-8").strip()
        return text or None
    return None


_KOKORO_VOICE_RE = re.compile(r"^[abefhijpz][fm]_")


def _is_kokoro_voice(voice: str) -> bool:
    return bool(voice) and bool(_KOKORO_VOICE_RE.match(voice))


# Orpheus ships eight named English voices; "tara" (first) is the default.
_ORPHEUS_VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")


def _is_orpheus_voice(voice: str) -> bool:
    return bool(voice) and voice.lower() in _ORPHEUS_VOICES


def _clone_voices() -> list[str]:
    """Reference clips in voices/ usable by the cloning backends."""
    if not VOICES_DIR.exists():
        return []
    return sorted({p.stem for p in VOICES_DIR.iterdir()
                   if p.suffix in (".wav", ".mp3", ".flac", ".m4a")})


_kokoro_voices_cache: dict[str, list[str]] = {}


def _kokoro_voices(repo: str) -> list[str]:
    """Named voices a Kokoro repo ships, from its voices/*.safetensors (cached)."""
    if repo not in _kokoro_voices_cache:
        try:
            from huggingface_hub import list_repo_files
            _kokoro_voices_cache[repo] = sorted(
                f[len("voices/"):-len(".safetensors")]
                for f in list_repo_files(repo)
                if f.startswith("voices/") and f.endswith(".safetensors"))
        except Exception:
            _kokoro_voices_cache[repo] = []
    return _kokoro_voices_cache[repo]


def _voices_for(model_id: str) -> dict:
    """Voices the given model accepts, plus whether it supports cloning."""
    backend = _backend(model_id)
    clones = _clone_voices()
    if backend == "kokoro":
        return {"backend": backend, "cloning": False,
                "voices": _kokoro_voices(model_id) or ["af_heart"]}
    if backend == "orpheus":
        return {"backend": backend, "cloning": True,
                "voices": list(_ORPHEUS_VOICES) + clones}
    return {"backend": backend, "cloning": True, "voices": ["default"] + clones}


def synthesize(req: SpeechRequest) -> tuple[bytes, str, float]:
    st = get_status()
    model = _state.get("model")
    if model is None or st["state"] != "ready":
        detail = (f"model is loading ({st['model']}); retry shortly"
                  if st["state"] == "loading"
                  else (st.get("error") or "model not loaded yet"))
        raise HTTPException(503, detail)
    if not req.input.strip():
        raise HTTPException(400, "input is empty")

    ref_audio = resolve_voice(req.voice)
    lang = (req.language or "").strip().lower()
    lang_code = lang if lang and lang != "unknown" else "en"
    kwargs: dict = {"speed": req.speed}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature

    if IS_KOKORO:
        # Kokoro: named voices (e.g. "af_heart"), no cloning. Map ISO language to
        # Kokoro's single-letter code; unknown voices fall back to the default.
        kwargs.pop("temperature", None)
        voice_label = req.voice if _is_kokoro_voice(req.voice) else "af_heart"
        kwargs["lang_code"] = voice_label[0]
        kwargs["voice"] = voice_label
        if ref_audio and not _is_kokoro_voice(req.voice):
            print(f"[speech] note: Kokoro has no cloning; '{req.voice}' is not a "
                  f"Kokoro voice. Using '{voice_label}'.")
    elif IS_FISH:
        # Fish/OpenAudio: no lang_code. Cloning needs the reference waveform as an
        # mx.array plus its transcript; without a transcript fall back to default.
        voice_label = "default"
        if ref_audio:
            ref_text = req.ref_text or resolve_ref_text(req.voice)
            if ref_text:
                from mlx_audio.utils import load_audio
                kwargs["ref_audio"] = load_audio(ref_audio, sample_rate=model.sample_rate)
                kwargs["ref_text"] = ref_text
                voice_label = req.voice
            else:
                print(f"[speech] note: no transcript for '{req.voice}'; add "
                      f"voices/{req.voice}.txt or send ref_text to clone. "
                      f"Using default voice.")
    elif IS_ORPHEUS:
        # Orpheus: named voices (tara, leah, ...), no lang_code. Cloning is
        # optional via a reference clip + its transcript; otherwise pick a named
        # voice, defaulting to "tara".
        kwargs.pop("speed", None)
        voice_label = req.voice if _is_orpheus_voice(req.voice) else "tara"
        kwargs["voice"] = voice_label
        if ref_audio:
            ref_text = req.ref_text or resolve_ref_text(req.voice)
            if ref_text:
                kwargs["ref_audio"] = ref_audio
                kwargs["ref_text"] = ref_text
                voice_label = req.voice
            else:
                print(f"[speech] note: no transcript for '{req.voice}'; add "
                      f"voices/{req.voice}.txt or send ref_text to clone. "
                      f"Using '{voice_label}'.")
    else:
        kwargs["lang_code"] = lang_code
        voice_label = req.voice if ref_audio else "default"
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
        if req.exaggeration is not None:
            kwargs["exaggeration"] = req.exaggeration
        if req.cfg_weight is not None:
            kwargs["cfg_weight"] = req.cfg_weight

    preview = req.input.strip().replace("\n", " ")
    if len(preview) > 60:
        preview = preview[:57] + "..."
    print(f"[speech] model={MODEL_ID} voice={voice_label} fmt={req.response_format} "
          f"lang={lang_code} chars={len(req.input)} text={preview!r}")

    gen_start = time.perf_counter()
    chunks: list[np.ndarray] = []
    sr = 24000
    for result in model.generate(text=req.input, verbose=False, **kwargs):
        sr = result.sample_rate or sr
        chunks.append(np.asarray(result.audio, dtype=np.float32))
    if not chunks:
        raise HTTPException(500, "no audio produced")
    gen_secs = time.perf_counter() - gen_start

    samples = np.concatenate(chunks)
    duration = len(samples) / sr
    encode_start = time.perf_counter()
    audio, content_type = encode_audio(samples, sr, req.response_format)
    encode_secs = time.perf_counter() - encode_start

    rtf = gen_secs / duration if duration else 0.0
    print(f"[speech] done audio={duration:.2f}s gen={gen_secs:.2f}s "
          f"encode={encode_secs:.2f}s rtf={rtf:.2f}x bytes={len(audio)}")
    return audio, content_type, duration


def _check_auth(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


# ---- routes ----------------------------------------------------------------

class LoadModelRequest(BaseModel):
    model: str  # a catalog key ("chatterbox", "openaudio") or a full MLX repo id


def _resolve_model_id(name: str) -> str:
    name = (name or "").strip()
    return KNOWN_MODELS.get(name.lower(), name)


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, authorization: str | None = Header(default=None)):
    # Dispatch generation to the dedicated MLX worker thread (same thread that
    # loaded the model) so the GPU stream is valid. Awaiting keeps the event loop
    # free to answer /health etc. while audio renders.
    _check_auth(authorization)
    st = get_status()
    if st["state"] != "ready" or "model" not in _state:
        detail = (f"model is loading ({st['model']}); retry shortly"
                  if st["state"] == "loading"
                  else (st.get("error") or "model not loaded yet"))
        raise HTTPException(503, detail)
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
    st = get_status()
    return {"object": "list", "active": st["model"], "state": st["state"], "data": [
        {"id": m["key"], "label": m["label"], "repo": m["repo"],
         "object": "model", "owned_by": "local", "active": st["model"] == m["repo"]}
        for m in MODEL_CATALOG
    ]}


@app.post("/v1/models/load")
async def load_model_route(req: LoadModelRequest, authorization: str | None = Header(default=None)):
    """Switch the active model. Returns immediately; poll /health for readiness."""
    _check_auth(authorization)
    model_id = _resolve_model_id(req.model)
    if not model_id:
        raise HTTPException(400, "model is required")
    st = get_status()
    if st["state"] == "loading":
        raise HTTPException(409, f"already loading {st['model']}; retry when ready")
    if st["state"] == "ready" and st["model"] == model_id:
        return {"state": "ready", "model": model_id, "changed": False}
    _set_status(state="loading", model=model_id, error=None)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_mlx, _load_model_on_worker, model_id)
    return {"state": "loading", "model": model_id, "changed": True}


@app.get("/v1/audio/voices")
def list_voices(model: str | None = None):
    """Voices for a model. Defaults to the active model; pass ?model=<key|repo>
    to preview another backend's voices without switching."""
    model_id = _resolve_model_id(model) if model else MODEL_ID
    return {"model": model_id, **_voices_for(model_id)}


@app.get("/health")
def health():
    st = get_status()
    return {"status": "ok", "state": st["state"], "model": st["model"],
            "backend": _backend(st["model"]),
            "loaded": st["state"] == "ready" and "model" in _state,
            "error": st["error"]}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TTS_HOST", "127.0.0.1")
    port = int(os.environ.get("TTS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
