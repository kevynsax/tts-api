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
    {"key": "higgs", "label": "Higgs Audio v2", "repo": "mlx-community/higgs-audio-v2-3B-mlx-q6"},
]
KNOWN_MODELS = {m["key"]: m["repo"] for m in MODEL_CATALOG}

# Fixed default seed so every stochastic sampler reproduces the same voice across
# requests (each sentence reads in one consistent voice). Kokoro is deterministic and
# unaffected. Override per-request with SpeechRequest.seed.
_DEFAULT_SEED = 42

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


def _is_higgs(model_id: str) -> bool:
    return "higgs" in model_id.lower()


def _backend(model_id: str) -> str:
    if _is_kokoro(model_id):
        return "kokoro"
    if _is_fish(model_id):
        return "fish"
    if _is_orpheus(model_id):
        return "orpheus"
    if _is_higgs(model_id):
        return "higgs"
    return "chatterbox"


IS_FISH = _is_fish(MODEL_ID)
IS_KOKORO = _is_kokoro(MODEL_ID)
IS_ORPHEUS = _is_orpheus(MODEL_ID)
IS_HIGGS = _is_higgs(MODEL_ID)
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


def _patch_fish_snake() -> None:
    """The Fish DAC vocoder's Snake activation evaluates sin(alpha*x) in bf16,
    quantizing the phase and adding metallic high-frequency artifacts. Compute it
    in float32; the rest of the codec stays bf16."""
    try:
        from mlx_audio.codec.models.fish_s1_dac import fish_s1_dac as _dac
        import mlx.core as mx
    except Exception:
        return
    if getattr(_dac, "_tts_snake_patched", False):
        return

    def _snake(x, alpha):
        xf = x.astype(mx.float32)
        af = alpha.astype(mx.float32)
        out = xf + mx.reciprocal(af + 1e-9) * mx.power(mx.sin(af * xf), 2)
        return out.astype(x.dtype)

    _dac.snake = _snake
    _dac._tts_snake_patched = True
    print("[server] applied fish snake fp32 patch")


def _load_model_on_worker(model_id: str) -> None:
    global MODEL_ID, IS_FISH, IS_KOKORO, IS_ORPHEUS, IS_HIGGS
    from mlx_audio.tts.utils import load_model
    backend = _backend(model_id)
    if backend == "kokoro":
        _patch_kokoro_vocoder()
    if backend == "fish":
        _patch_fish_snake()
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
    IS_HIGGS = _is_higgs(model_id)
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
    # RNG seed. Fish/OpenAudio sampling is stochastic, so without a fixed seed the
    # voice drifts between requests; defaults to a fixed seed for Fish so the same
    # voice stays consistent. Pass an int to override, or for any model.
    seed: int | None = None
    # Trim per-sentence leading/trailing silence so concatenated sentences butt up
    # tightly (no gaps, no ambient-noise bursts at the joins). gap_ms inserts a fixed
    # pause between sentences after trimming; 0 = none.
    # denoise applies a stationary spectral-gate noise reducer to the rendered audio.
    # trim_silence/denoise default to None = auto: on for Orpheus, off elsewhere.
    trim_silence: bool | None = None
    gap_ms: int = 0
    denoise: bool | None = None
    denoise_amount: float = 0.9


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


def _trim_silence(
    samples: np.ndarray, sr: int, head_pad_ms: int = 150, tail_pad_ms: int = 450
) -> np.ndarray:
    """Trim leading/trailing silence (and the ambient noise floor that lives in it).

    Keeps everything above a peak-relative threshold plus a small absolute floor,
    padded generously so word onsets/decays are never clipped. The tail pad is much
    larger than the head pad so each sentence keeps room to breathe afterwards. Only
    the head and tail are touched; pauses inside the sentence are left intact.
    """
    if samples.size == 0:
        return samples
    # Smoothed RMS envelope over ~30 ms so isolated spikes (residual spectral-gate
    # noise) don't count as speech and keep the silence from being trimmed.
    win = max(1, int(sr * 0.03))
    power = samples.astype(np.float32) ** 2
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.sqrt(np.convolve(power, kernel, mode="same"))
    peak = float(env.max())
    if peak <= 0:
        return samples[:0]
    thr = max(peak * 0.004, 0.0015)
    loud = np.flatnonzero(env > thr)
    if loud.size == 0:
        return samples[:0]
    head_pad = int(sr * head_pad_ms / 1000)
    tail_pad = int(sr * tail_pad_ms / 1000)
    start = max(0, int(loud[0]) - head_pad)
    end = min(samples.size, int(loud[-1]) + 1 + tail_pad)
    return samples[start:end]


_df_state: dict = {}


def _get_df():
    """Lazily load DeepFilterNet3 (cached). Returns (model, df_state) or None.

    deepfilternet 0.5.6 imports torchaudio.backend.common.AudioMetaData, removed in
    recent torchaudio; we feed tensors straight to enhance() and never use df.io, so
    a tiny shim lets the package import cleanly.
    """
    if "loaded" in _df_state:
        return _df_state["loaded"]
    try:
        import sys
        import types
        import torchaudio
        if "torchaudio.backend.common" not in sys.modules:
            bc = types.ModuleType("torchaudio.backend.common")
            bc.AudioMetaData = type("AudioMetaData", (), {})
            b = types.ModuleType("torchaudio.backend")
            b.common = bc
            sys.modules["torchaudio.backend"] = b
            sys.modules["torchaudio.backend.common"] = bc
            torchaudio.backend = b
        from df.enhance import init_df
        model, state, _ = init_df()
        _df_state["loaded"] = (model, state)
        print("[server] loaded DeepFilterNet3 denoiser")
    except Exception as e:
        _df_state["loaded"] = None
        print(f"[server] DeepFilterNet unavailable ({e}); falling back to noisereduce")
    return _df_state["loaded"]


def _denoise(samples: np.ndarray, sr: int, amount: float = 0.9) -> np.ndarray:
    """Strip the model's ambient hum. Prefers DeepFilterNet3 (learned, artifact-free);
    falls back to a stationary spectral gate if it can't load.

    amount (0..1) eases off the suppression: 1.0 = full, lower keeps more of the
    original (DeepFilterNet caps attenuation in dB; noisereduce scales prop_decrease).
    """
    if samples.size == 0 or amount <= 0:
        return samples
    amount = float(np.clip(amount, 0.0, 1.0))
    df = _get_df()
    if df is not None:
        import torch
        from scipy.signal import resample_poly
        model, state = df
        dsr = state.sr()
        up = resample_poly(samples, dsr, sr).astype(np.float32) if sr != dsr else samples
        from df.enhance import enhance
        atten = None if amount >= 0.999 else amount * 60.0
        out = enhance(model, state, torch.from_numpy(np.ascontiguousarray(up)).unsqueeze(0),
                      atten_lim_db=atten)
        out = out.squeeze(0).cpu().numpy()
        if sr != dsr:
            out = resample_poly(out, sr, dsr)
        return np.asarray(out, dtype=np.float32)
    import noisereduce as nr
    out = nr.reduce_noise(y=samples, sr=sr, stationary=True, prop_decrease=amount)
    return np.asarray(out, dtype=np.float32)


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


# Fish/OpenAudio and Higgs sample a brand-new speaker whenever they get no reference
# clip, so the "default" voice drifts between requests even with a fixed seed (the RNG
# stream depends on the text). Pin it by cloning every default-voice request from an
# anchor clip in voices/auto/. The anchors are committed to the repo so every
# deployment (MLX or CPU, any machine) uses the same default voice; if one is missing
# it is regenerated once with a fixed seed, but that regenerated voice is local to
# the deployment until committed.
_ANCHOR_DIR = VOICES_DIR / "auto"
_ANCHOR_TEXT = ("Here is a quick sample of my default voice reading aloud in the "
                "same clear and steady tone that I will always use for you.")


def _tighten_anchor(samples: np.ndarray, sr: int) -> np.ndarray:
    """Trim edge silence and collapse internal pauses to ~250 ms. A reference clip
    with long dead air teaches the model to emit silence until the token budget runs
    out, so the anchor must be tight."""
    win = max(1, int(sr * 0.03))
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.sqrt(np.convolve(samples.astype(np.float32) ** 2, kernel, mode="same"))
    thr = max(float(env.max()) * 0.004, 0.0015)
    loud = np.flatnonzero(env > thr)
    if loud.size == 0:
        return samples
    max_gap = int(sr * 0.25)
    pieces, start, prev = [], int(loud[0]), int(loud[0])
    for i in map(int, loud[1:]):
        if i - prev > max_gap:
            pieces.append(samples[start:prev + max_gap])
            start = i
        prev = i
    pieces.append(samples[start:prev + 1])
    return np.concatenate(pieces)


def _default_anchor(model) -> tuple[str, str] | None:
    """Return (wav_path, transcript) of the cached default-voice reference clip,
    generating it on first use. None if generation fails (caller falls back to the
    unanchored default)."""
    wav = _ANCHOR_DIR / f"{_backend(MODEL_ID)}.wav"
    txt = wav.with_suffix(".txt")
    if wav.exists() and txt.exists():
        return str(wav), txt.read_text(encoding="utf-8").strip()
    try:
        import mlx.core as mx
        mx.random.seed(_DEFAULT_SEED)
        chunks, sr = [], 24000
        print("[speech] building default-voice anchor clip (one-time). Note: this "
              "clip is local to this deployment; commit voices/auto/ to keep the "
              "default voice identical across servers.")
        for result in model.generate(text=_ANCHOR_TEXT, verbose=False):
            sr = result.sample_rate or sr
            chunk = np.asarray(result.audio, dtype=np.float32)
            if chunk.size:
                chunks.append(chunk)
        if not chunks:
            return None
        _ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav), _tighten_anchor(np.concatenate(chunks), sr), sr)
        txt.write_text(_ANCHOR_TEXT, encoding="utf-8")
        return str(wav), _ANCHOR_TEXT
    except Exception as e:
        print(f"[speech] note: default-voice anchor failed ({e}); "
              f"default voice may vary between calls.")
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


def _model_key(model_id: str) -> str:
    """Catalog key for a model id/repo (used to pick its voice-name set)."""
    for m in MODEL_CATALOG:
        if model_id in (m["repo"], m["key"]):
            return m["key"]
    return _backend(model_id)


# Each model presents the SAME underlying clip under a different real first name, so
# the UI shows distinct names per model and the name alone tells you which model made
# a clip. Names keep the country/gender of the source voice (GB, US, BR, PT).
VOICE_ALIASES: dict[str, dict[str, str]] = {
    "chatterbox": {
        "default": "Oliver",
        "en-GB-RyanNeural": "Arthur", "en-GB-SoniaNeural": "Eleanor",
        "en-US-AndrewNeural": "Samuel", "en-US-AvaNeural": "Rachel",
        "en-US-BrianNeural": "Henry", "en-US-EmmaNeural": "Ruth",
        "pt-BR-AntonioNeural": "Mateus", "pt-BR-FranciscaNeural": "Helena",
        "pt-BR-ThalitaMultilingualNeural": "Beatriz",
        "pt-PT-DuarteNeural": "Tomás", "pt-PT-RaquelNeural": "Inês",
    },
    "openaudio": {
        "default": "Theodore",
        "en-GB-RyanNeural": "Edward", "en-GB-SoniaNeural": "Charlotte",
        "en-US-AndrewNeural": "Nathan", "en-US-AvaNeural": "Grace",
        "en-US-BrianNeural": "Walter", "en-US-EmmaNeural": "Naomi",
        "pt-BR-AntonioNeural": "Rafael", "pt-BR-FranciscaNeural": "Larissa",
        "pt-BR-ThalitaMultilingualNeural": "Camila",
        "pt-PT-DuarteNeural": "Gonçalo", "pt-PT-RaquelNeural": "Matilde",
    },
    "higgs": {
        "default": "Julian",
        "en-GB-RyanNeural": "Sebastian", "en-GB-SoniaNeural": "Imogen",
        "en-US-AndrewNeural": "Caleb", "en-US-AvaNeural": "Vivian",
        "en-US-BrianNeural": "Gordon", "en-US-EmmaNeural": "Hazel",
        "pt-BR-AntonioNeural": "Bruno", "pt-BR-FranciscaNeural": "Renata",
        "pt-BR-ThalitaMultilingualNeural": "Bianca",
        "pt-PT-DuarteNeural": "Afonso", "pt-PT-RaquelNeural": "Carolina",
    },
    "orpheus": {
        "tara": "Sophie", "leah": "Diana", "jess": "Megan", "leo": "Marcus",
        "dan": "Victor", "mia": "Paula", "zac": "Derek", "zoe": "Tessa",
    },
    "kokoro": {
        "af_heart": "Hannah", "af_bella": "Bella", "af_nicole": "Nicole",
        "af_sarah": "Sarah", "af_sky": "Skyler", "af_alloy": "Allison",
        "af_aoede": "Audrey", "af_jessica": "Jessica", "af_kore": "Cora",
        "af_nova": "Nora", "af_river": "Riley",
        "am_adam": "Adam", "am_michael": "Michael", "am_echo": "Elliot",
        "am_eric": "Eric", "am_fenrir": "Fenton", "am_liam": "Liam",
        "am_onyx": "Owen", "am_puck": "Parker", "am_santa": "Nicholas",
        "bf_emma": "Emily", "bf_alice": "Alice", "bf_isabella": "Isabella",
        "bf_lily": "Lily", "bm_daniel": "Daniel", "bm_george": "George",
        "bm_lewis": "Lewis", "bm_fable": "Felix",
    },
}

_DERIVE_KOKORO = re.compile(r"^[a-z]{2}_(.+)$")
_DERIVE_AZURE = re.compile(r"^[a-z]{2}-[A-Z]{2}-(.+?)(?:Multilingual)?Neural$")


def _derive_name(raw: str) -> str:
    """Fallback display name for a voice with no curated alias."""
    m = _DERIVE_AZURE.match(raw)
    if m:
        return m.group(1)
    m = _DERIVE_KOKORO.match(raw)
    if m:
        return m.group(1).replace("_", " ").title()
    return raw[:1].upper() + raw[1:] if raw else raw


def _display_voice(model_id: str, raw: str) -> str:
    return VOICE_ALIASES.get(_model_key(model_id), {}).get(raw) or _derive_name(raw)


def _raw_voices_for(model_id: str) -> tuple[list[str], bool]:
    """Underlying voice ids a model accepts, plus whether it supports cloning."""
    backend = _backend(model_id)
    clones = _clone_voices()
    if backend == "kokoro":
        return (_kokoro_voices(model_id) or ["af_heart"]), False
    if backend == "orpheus":
        return (list(_ORPHEUS_VOICES) + clones), True
    return (["default"] + clones), True  # chatterbox, fish/openaudio, higgs


def _resolve_voice_name(model_id: str, name: str) -> str:
    """Map a display name back to its underlying voice id. Raw ids and unknown names
    pass through unchanged (so old clients keep working)."""
    if not name:
        return name
    low = name.strip().lower()
    raw, _ = _raw_voices_for(model_id)
    for v in raw:
        if _display_voice(model_id, v).lower() == low:
            return v
    return name


def _voices_for(model_id: str) -> dict:
    """Voices a model accepts. `voices` keeps the underlying ids (so clients can group
    them by language/locale); `names` maps each id to its model-specific display name.
    Requests may send either the id or the display name."""
    raw, cloning = _raw_voices_for(model_id)
    return {"backend": _backend(model_id), "cloning": cloning,
            "voices": raw,
            "names": {v: _display_voice(model_id, v) for v in raw}}


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

    req.voice = _resolve_voice_name(MODEL_ID, req.voice)
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
        if "ref_audio" not in kwargs:
            anchor = _default_anchor(model)
            if anchor:
                from mlx_audio.utils import load_audio
                kwargs["ref_audio"] = load_audio(anchor[0], sample_rate=model.sample_rate)
                kwargs["ref_text"] = anchor[1]
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
    elif IS_HIGGS:
        # Higgs Audio v2: no named voices and no lang_code. Cloning is optional via
        # a reference clip + its transcript; the reference must be a mono 24 kHz
        # array. Without a reference it runs "smart voice" mode.
        kwargs.pop("speed", None)
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
                      f"Using smart-voice default.")
        if "ref_audio" not in kwargs:
            anchor = _default_anchor(model)
            if anchor:
                from mlx_audio.utils import load_audio
                kwargs["ref_audio"] = load_audio(anchor[0], sample_rate=model.sample_rate)
                kwargs["ref_text"] = anchor[1]
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

    do_trim = req.trim_silence if req.trim_silence is not None else IS_ORPHEUS
    do_denoise = req.denoise if req.denoise is not None else IS_ORPHEUS

    # Pin the RNG so a voice is reproducible across requests. Every model's sampler is
    # stochastic (except deterministic Kokoro) and otherwise picks a different voice
    # each call, so they default to a fixed seed unless the request asks otherwise.
    seed = req.seed if req.seed is not None else _DEFAULT_SEED
    if seed is not None:
        import mlx.core as mx
        mx.random.seed(int(seed))

    gen_start = time.perf_counter()
    chunks: list[np.ndarray] = []
    sr = 24000
    for result in model.generate(text=req.input, verbose=False, **kwargs):
        sr = result.sample_rate or sr
        audio_chunk = np.asarray(result.audio, dtype=np.float32)
        # Denoise before trimming: the ambient floor otherwise reads as "loud" and
        # defeats the silence detector, so the gaps wouldn't shrink.
        if do_denoise:
            audio_chunk = _denoise(audio_chunk, sr, req.denoise_amount)
        if do_trim:
            audio_chunk = _trim_silence(audio_chunk, sr)
        if audio_chunk.size:
            chunks.append(audio_chunk)
    if not chunks:
        raise HTTPException(500, "no audio produced")
    gen_secs = time.perf_counter() - gen_start

    if do_trim and req.gap_ms > 0:
        gap = np.zeros(int(sr * req.gap_ms / 1000), dtype=np.float32)
        joined: list[np.ndarray] = []
        for i, c in enumerate(chunks):
            if i:
                joined.append(gap)
            joined.append(c)
        chunks = joined
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
