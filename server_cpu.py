#!/usr/bin/env python
"""
Containerizable OpenAI-compatible TTS server (PyTorch) for Linux / x86 / k8s.

Mirrors the native MLX server.py HTTP API, but uses PyTorch backends because MLX
needs Apple Silicon + Metal and cannot run in a Linux container. Runs on CPU or
CUDA (auto-detected). The native server.py remains the fast path on the Mac host.

Backends (selectable at runtime):
    chatterbox  -> chatterbox-tts (PyTorch), clones from a reference clip
    kokoro      -> kokoro (PyTorch), small/fast, named voices (e.g. af_heart)
    orpheus     -> orpheus-3b (PyTorch) + SNAC, named voices, GPU
    higgs       -> Higgs Audio v2 (PyTorch), GPU; 4-bit quantized by default,
                   optional voice cloning via a reference clip + transcript
  (OpenAudio/Fish is MLX-host-only; fish-speech has no clean pip inference API.)

Routes (same shape as server.py):
    POST /v1/audio/speech    {model, input, voice, response_format, speed, language}
    GET  /v1/models          list backends + which is active
    POST /v1/models/load     hot-swap the active backend (returns immediately)
    GET  /v1/audio/voices
    GET  /health             liveness + load state (loading|ready|error)
    GET  /ready              200 only when the model is in memory (k8s readiness)

Memory:
    By default the model loads on the first request and unloads after idle, so the
    GPU/CPU stays free for other work when the server isn't in use. The first
    request after an idle period pays a cold-start reload.

Env:
    TTS_MODEL    default backend key ("chatterbox" | "kokoro")
    TTS_DEVICE / CHATTERBOX_TORCH_DEVICE   auto | cpu | cuda | mps
    TTS_IDLE_UNLOAD_SECONDS   unload after N idle seconds (default 300; 0 = never)
    TTS_PRELOAD  "1" to load at startup instead of on first request
    TTS_HOST (0.0.0.0), TTS_PORT (8000), TTS_API_KEY (optional bearer auth)
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# Let unsupported ops on Apple's MPS backend (e.g. iSTFTNet's istft/complex math)
# fall back to CPU instead of crashing. Harmless on non-Mac. Must be set before torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from setproctitle import setproctitle
    setproctitle("tts-server")
except Exception:
    pass

KOKORO_REPO = os.environ.get("KOKORO_REPO", "hexgrad/Kokoro-82M")
# Default to the ungated unsloth mirror so the container loads without an HF token;
# canopylabs/orpheus-3b-0.1-ft is the gated upstream (set ORPHEUS_REPO + HF_TOKEN).
ORPHEUS_REPO = os.environ.get("ORPHEUS_REPO", "unsloth/orpheus-3b-0.1-ft")
ORPHEUS_SNAC_REPO = os.environ.get("ORPHEUS_SNAC_REPO", "hubertsiuzdak/snac_24khz")

# Higgs Audio v2 (PyTorch, native transformers >=5.3). The bf16 3B wants ~24 GB
# VRAM, so default to 4-bit bitsandbytes (NF4) quantization (~5 GB; 8-bit ~7 GB;
# 0 = full bf16). Quantization needs CUDA + bitsandbytes; on CPU it loads full
# precision. The eustlb mirror ships the transformers processor + chat template.
HIGGS_REPO = os.environ.get("HIGGS_REPO", "eustlb/higgs-audio-v2-generation-3B-base")
HIGGS_QUANT_BITS = int(os.environ.get("HIGGS_QUANT_BITS", "4"))
HIGGS_MAX_NEW_TOKENS = int(os.environ.get("HIGGS_MAX_NEW_TOKENS", "2048"))
# Long input in one generate() call overflows the model's token budget and the audio is
# truncated; split it into sentence-sized chunks rendered separately and concatenated.
HIGGS_MAX_CHARS = int(os.environ.get("HIGGS_MAX_CHARS", "350"))
HIGGS_SCENE = os.environ.get("HIGGS_SCENE", "Audio is recorded from a quiet room.")
# Fixed default seed so every stochastic sampler (Chatterbox, Orpheus, Higgs) reproduces
# the same voice across requests. Kokoro is deterministic and unaffected. Override
# per-request with SpeechRequest.seed.
_DEFAULT_SEED = 42

# Switchable backends. `key` is what clients pass; everything PyTorch-based here.
MODEL_CATALOG = [
    {"key": "chatterbox", "label": "Chatterbox", "backend": "chatterbox"},
    {"key": "kokoro", "label": "Kokoro", "backend": "kokoro"},
    {"key": "orpheus", "label": "Orpheus", "backend": "orpheus"},
    {"key": "higgs", "label": "Higgs Audio v2", "backend": "higgs"},
]
KNOWN_KEYS = {m["key"] for m in MODEL_CATALOG}

MODEL_KEY = os.environ.get("TTS_MODEL", "chatterbox").strip().lower()
if MODEL_KEY not in KNOWN_KEYS:
    MODEL_KEY = "chatterbox"

VOICES_DIR = Path(__file__).parent / "voices"
API_KEY = os.environ.get("TTS_API_KEY")

# Keep the GPU/CPU free for other work: load on first request, then unload after
# this many idle seconds (0 disables auto-unload). TTS_PRELOAD=1 loads at startup
# instead of waiting for the first request.
IDLE_UNLOAD_SECONDS = int(os.environ.get("TTS_IDLE_UNLOAD_SECONDS", "300"))
PRELOAD = os.environ.get("TTS_PRELOAD", "0").strip().lower() in ("1", "true", "yes")

# Serialize all model work (load + generate + unload) on one worker; also lets the
# load run in the background so /health answers immediately with state="loading".
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
_state: dict = {}

# Wall-clock of the last request, so the idle monitor knows when to unload.
_activity_lock = threading.Lock()
_last_activity = 0.0
_unload_pending = False

_status_lock = threading.Lock()
# "idle" = nothing in memory but ready to load on demand; "loading"/"ready"/"error".
_status: dict = {"state": "loading" if PRELOAD else "idle", "model": MODEL_KEY, "error": None}


def _set_status(**kw) -> None:
    with _status_lock:
        _status.update(kw)


def get_status() -> dict:
    with _status_lock:
        return dict(_status)


def _touch() -> None:
    global _last_activity
    with _activity_lock:
        _last_activity = time.monotonic()


def _free_device_memory(device: str) -> None:
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _unload_on_worker() -> None:
    """Drop the model and hand its GPU/CPU memory back. Runs on the worker thread,
    so it can never overlap a generation."""
    global _unload_pending
    _unload_pending = False
    loaded = _state.pop("loaded", None)
    if loaded is None:
        return
    device = loaded.get("device", "cpu")
    loaded.clear()
    import gc
    gc.collect()
    _free_device_memory(device)
    _set_status(state="idle", model=MODEL_KEY, error=None)
    print(f"[server-cpu] unloaded {MODEL_KEY}; freed {device} memory")


def _idle_monitor() -> None:
    """Background loop: unload the model after IDLE_UNLOAD_SECONDS of inactivity."""
    global _unload_pending
    tick = max(5, min(IDLE_UNLOAD_SECONDS, 30))
    while True:
        time.sleep(tick)
        if "loaded" not in _state or _unload_pending:
            continue
        with _activity_lock:
            idle = time.monotonic() - _last_activity
        if idle >= IDLE_UNLOAD_SECONDS:
            _unload_pending = True
            _worker.submit(_unload_on_worker)


def pick_device() -> str:
    pref = (os.environ.get("TTS_DEVICE")
            or os.environ.get("CHATTERBOX_TORCH_DEVICE", "auto")).lower()
    if pref != "auto":
        return pref
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---- model loading ---------------------------------------------------------

def _load_chatterbox(device: str) -> dict:
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=device)
    return {"backend": "chatterbox", "model": model, "sr": int(model.sr)}


def _load_kokoro(device: str) -> dict:
    from kokoro import KModel, KPipeline
    kmodel = KModel(repo_id=KOKORO_REPO)
    if device in ("cuda", "mps"):
        kmodel = kmodel.to(device)
    kmodel.eval()
    # Pipelines are language-specific; cache one per language code, sharing weights.
    pipes = {"a": KPipeline(lang_code="a", repo_id=KOKORO_REPO, model=kmodel)}
    return {"backend": "kokoro", "kmodel": kmodel, "pipes": pipes, "sr": 24000}


def _load_orpheus(device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(ORPHEUS_REPO, torch_dtype=dtype)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_REPO)
    snac = SNAC.from_pretrained(ORPHEUS_SNAC_REPO).eval().to(device)
    return {"backend": "orpheus", "model": model, "tokenizer": tokenizer,
            "snac": snac, "sr": 24000}


def _load_higgs(device: str) -> dict:
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import HiggsAudioV2ForConditionalGeneration as HiggsGen
    except ImportError:  # older naming before the _v2 rename
        from transformers import HiggsAudioForConditionalGeneration as HiggsGen

    processor = AutoProcessor.from_pretrained(HIGGS_REPO)
    kwargs: dict = {}
    if HIGGS_QUANT_BITS in (4, 8) and device == "cuda":
        from transformers import BitsAndBytesConfig
        if HIGGS_QUANT_BITS == 4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": device}
        print(f"[server-cpu] Higgs loading in {HIGGS_QUANT_BITS}-bit quantization")
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = HiggsGen.from_pretrained(HIGGS_REPO, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()

    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None and hasattr(audio_tokenizer, "to"):
        audio_tokenizer.to(device)

    return {"backend": "higgs", "model": model, "processor": processor, "sr": 24000}


_BACKEND_LOADERS = {"chatterbox": _load_chatterbox, "kokoro": _load_kokoro,
                    "orpheus": _load_orpheus, "higgs": _load_higgs}


def _load_model_on_worker(model_key: str) -> None:
    global MODEL_KEY
    backend = next((m["backend"] for m in MODEL_CATALOG if m["key"] == model_key), None)
    if backend is None:
        _set_status(state="error", model=model_key, error=f"unknown model '{model_key}'")
        return
    if backend not in _BACKEND_LOADERS:
        _set_status(state="error", model=model_key,
                    error=f"'{model_key}' is MLX-only and not supported on the CPU server")
        return
    _set_status(state="loading", model=model_key, error=None)
    device = pick_device()
    print(f"[server-cpu] loading {model_key} (backend={backend}) on {device} ...")
    try:
        loaded = _BACKEND_LOADERS[backend](device)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"[server-cpu] load FAILED for {model_key}: {msg}")
        if _state.get("loaded"):
            _set_status(state="ready", model=MODEL_KEY, error=msg)  # kept previous model
        else:
            _set_status(state="error", model=model_key, error=msg)
        return
    loaded["device"] = device
    _state["loaded"] = loaded
    MODEL_KEY = model_key
    _touch()
    print(f"[server-cpu] ffmpeg: {FFMPEG or 'NOT FOUND'}")
    print(f"[server-cpu] model ready: {model_key} on {device}")
    _set_status(state="ready", model=model_key, error=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    loop = asyncio.get_running_loop()
    if PRELOAD:
        loop.run_in_executor(_worker, _load_model_on_worker, MODEL_KEY)
    else:
        print(f"[server-cpu] lazy mode: '{MODEL_KEY}' loads on first request, "
              f"unloads after {IDLE_UNLOAD_SECONDS}s idle"
              if IDLE_UNLOAD_SECONDS else
              f"[server-cpu] lazy mode: '{MODEL_KEY}' loads on first request")
    if IDLE_UNLOAD_SECONDS > 0:
        threading.Thread(target=_idle_monitor, name="idle-monitor", daemon=True).start()
    yield
    _state.clear()
    _worker.shutdown(wait=False)


app = FastAPI(title="TTS Server (PyTorch, OpenAI-compatible)", version="1.0.0", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float = 1.0
    language: str | None = None
    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None  # Orpheus/Higgs sampling temperature
    ref_text: str | None = None  # Higgs voice-clone reference transcript
    # RNG seed: pin sampling so a voice reproduces across requests. (Fish/OpenAudio,
    # which most needs this, is MLX-only and lives in server.py.)
    seed: int | None = None


class LoadModelRequest(BaseModel):
    model: str  # catalog key: "chatterbox" | "kokoro"


# ---- audio helpers ---------------------------------------------------------

_CONTENT_TYPES = {
    "wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg",
    "flac": "audio/flac", "opus": "audio/ogg", "aac": "audio/aac",
}
_FFMPEG_FMT = {"mp3": "mp3", "flac": "flac", "opus": "opus", "aac": "adts"}
# High bitrates so lossy codecs don't add shimmer on top of the vocoder; flac is
# lossless (none), opus is efficient so 96k is transparent.
_FFMPEG_BITRATE = {"mp3": "256k", "aac": "256k", "opus": "96k"}


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for p in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
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
    if fmt == "pcm":
        return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes(), _CONTENT_TYPES["pcm"]
    if not FFMPEG:
        raise HTTPException(500, "ffmpeg not found; install it or request 'wav'/'pcm'.")
    wav = _wav_bytes(samples, sr)
    bitrate = ["-b:a", _FFMPEG_BITRATE[fmt]] if fmt in _FFMPEG_BITRATE else []
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", _FFMPEG_FMT[fmt], *bitrate, "pipe:1"],
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


def resolve_ref_text(voice: str) -> str | None:
    """Transcript for a cloned voice from voices/<voice>.txt, if present."""
    if not voice or voice in ("default", "chatterbox"):
        return None
    p = VOICES_DIR / f"{voice}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


_SENT_SPLIT = re.compile(r"(?<=[.!?…。！？])\s+|\n+")


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Greedily pack sentences into chunks no longer than max_chars, hard-wrapping any
    lone sentence that still overflows on a word boundary."""
    text = text.strip()
    if not text:
        return []
    chunks, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
        while len(cur) > max_chars:
            cut = cur.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(cur[:cut].strip())
            cur = cur[cut:].strip()
    if cur:
        chunks.append(cur)
    return chunks


def _higgs_generate(loaded: dict, text: str, ref_audio: str | None,
                    ref_text: str | None, temperature: float | None):
    """Render speech with Higgs Audio v2 (native transformers). With a reference clip
    + its transcript it zero-shot clones the voice; otherwise smart-voice default. Long
    input is split into chunks (each within the token budget) and concatenated so the
    audio isn't truncated; every chunk shares the reference, keeping one voice."""
    import tempfile
    import torch

    model = loaded["model"]
    processor = loaded["processor"]
    base = [
        {"role": "system",
         "content": [{"type": "text", "text": "Generate audio following instruction."}]},
        {"role": "scene", "content": [{"type": "text", "text": HIGGS_SCENE}]},
    ]
    if ref_audio and ref_text:
        base.append({"role": "user", "content": [{"type": "text", "text": ref_text}]})
        base.append({"role": "assistant",
                     "content": [{"type": "audio", "url": ref_audio}]})

    gen_kwargs: dict = {"max_new_tokens": HIGGS_MAX_NEW_TOKENS}
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs["do_sample"] = False

    segments: list[np.ndarray] = []
    sr = 24000
    for part in _chunk_text(text, HIGGS_MAX_CHARS) or [text]:
        conversation = base + [{"role": "user", "content": [{"type": "text", "text": part}]}]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, sampling_rate=24000, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        decoded = processor.batch_decode(outputs)
        # Let the processor reconstruct the waveform, then read it back as samples.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            processor.save_audio(decoded, tf.name)
            seg, sr = sf.read(tf.name, dtype="float32")
        if seg.ndim > 1:
            seg = seg.mean(axis=1)
        if seg.size:
            segments.append(np.asarray(seg, dtype=np.float32))

    if not segments:
        raise HTTPException(500, "no audio produced")
    if len(segments) == 1:
        return segments[0], int(sr)
    gap = np.zeros(int(sr * 0.15), dtype=np.float32)
    joined: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i:
            joined.append(gap)
        joined.append(seg)
    return np.concatenate(joined), int(sr)


def _clone_voices() -> list[str]:
    if not VOICES_DIR.exists():
        return []
    return sorted({p.stem for p in VOICES_DIR.iterdir()
                   if p.suffix in (".wav", ".mp3", ".flac", ".m4a")})


_kokoro_voices_cache: dict[str, list[str]] = {}


def _kokoro_repo_voices() -> list[str]:
    if KOKORO_REPO not in _kokoro_voices_cache:
        try:
            from huggingface_hub import list_repo_files
            _kokoro_voices_cache[KOKORO_REPO] = sorted(
                f[len("voices/"):-len(".safetensors")]
                for f in list_repo_files(KOKORO_REPO)
                if f.startswith("voices/") and f.endswith(".safetensors"))
        except Exception:
            _kokoro_voices_cache[KOKORO_REPO] = []
    return _kokoro_voices_cache[KOKORO_REPO]


# Each model presents the SAME underlying clip under a different real first name, so
# the UI shows distinct names per model and the name alone tells you which model made
# a clip. Names keep the country/gender of the source voice (GB, US, BR, PT). Kept in
# sync with server.py.
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


def _display_voice(key: str, raw: str) -> str:
    return VOICE_ALIASES.get(key, {}).get(raw) or _derive_name(raw)


def _raw_voices_for(key: str) -> tuple[list[str], bool]:
    """Underlying voice ids a model accepts, plus whether it supports cloning."""
    backend = next((m["backend"] for m in MODEL_CATALOG if m["key"] == key), "chatterbox")
    if backend == "kokoro":
        return (_kokoro_repo_voices() or ["af_heart"]), False
    if backend == "orpheus":  # named voices only; cloning is unreliable on Orpheus
        return list(_ORPHEUS_VOICES), False
    return (["default"] + _clone_voices()), True


def _resolve_voice_name(key: str, name: str) -> str:
    """Map a display name back to its underlying voice id. Raw ids and unknown names
    pass through unchanged (so old clients keep working)."""
    if not name:
        return name
    low = name.strip().lower()
    raw, _ = _raw_voices_for(key)
    for v in raw:
        if _display_voice(key, v).lower() == low:
            return v
    return name


def _voices_for(key: str) -> dict:
    """Voices a model accepts. `voices` keeps the underlying ids (so clients can group
    them by language/locale); `names` maps each id to its model-specific display name.
    Requests may send either the id or the display name."""
    backend = next((m["backend"] for m in MODEL_CATALOG if m["key"] == key), "chatterbox")
    raw, cloning = _raw_voices_for(key)
    return {"backend": backend, "cloning": cloning,
            "voices": raw,
            "names": {v: _display_voice(key, v) for v in raw}}


# ---- Kokoro helpers --------------------------------------------------------

_KOKORO_VOICE_RE = re.compile(r"^[abefhijpz][fm]_")


def _is_kokoro_voice(voice: str) -> bool:
    return bool(voice) and bool(_KOKORO_VOICE_RE.match(voice))


def _kokoro_pipeline(lang_letter: str):
    from kokoro import KPipeline
    loaded = _state["loaded"]
    pipes = loaded["pipes"]
    if lang_letter not in pipes:
        pipes[lang_letter] = KPipeline(lang_code=lang_letter, repo_id=KOKORO_REPO,
                                       model=loaded["kmodel"])
    return pipes[lang_letter]


def _to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


# ---- Orpheus helpers -------------------------------------------------------

# Orpheus ships eight named English voices; "tara" (first) is the default.
_ORPHEUS_VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
# Special tokens in the Orpheus prompt/codebook layout.
_ORPHEUS_SOH, _ORPHEUS_EOT, _ORPHEUS_EOH = 128259, 128009, 128260  # start human, end text, end human
_ORPHEUS_SOA, _ORPHEUS_EOA = 128257, 128258  # start / end of audio
_ORPHEUS_CODE_OFFSET = 128266               # first audio-code token id
_ORPHEUS_CODEBOOK = 4096                    # SNAC codebook size per slot


def _is_orpheus_voice(voice: str) -> bool:
    return bool(voice) and voice.lower() in _ORPHEUS_VOICES


def _orpheus_generate_codes(loaded: dict, text: str, voice: str, temperature: float) -> list[int]:
    """Run the Orpheus LM and return the flat list of SNAC code ids (offset removed)."""
    import torch
    model, tok, device = loaded["model"], loaded["tokenizer"], loaded["device"]
    prompt_ids = tok(f"{voice}: {text}", return_tensors="pt").input_ids
    start = torch.tensor([[_ORPHEUS_SOH]], dtype=torch.int64)
    end = torch.tensor([[_ORPHEUS_EOT, _ORPHEUS_EOH]], dtype=torch.int64)
    ids = torch.cat([start, prompt_ids, end], dim=1).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=1200, do_sample=True, temperature=max(temperature, 1e-4),
            top_p=0.8, repetition_penalty=1.3, eos_token_id=_ORPHEUS_EOA,
        )
    gen = out[0, ids.shape[1]:].tolist()
    # Crop to audio tokens (after the last start-of-audio), drop end markers,
    # trim to a whole number of 7-token frames, and undo the code offset.
    if _ORPHEUS_SOA in gen:
        gen = gen[len(gen) - 1 - gen[::-1].index(_ORPHEUS_SOA) + 1:]
    gen = [t for t in gen if t != _ORPHEUS_EOA]
    gen = gen[: (len(gen) // 7) * 7]
    return [t - _ORPHEUS_CODE_OFFSET for t in gen]


def _orpheus_decode(loaded: dict, codes: list[int]) -> np.ndarray:
    """Redistribute a flat code list into SNAC's 3 layers and decode to audio."""
    import torch
    cb = _ORPHEUS_CODEBOOK
    l1, l2, l3 = [], [], []
    for i in range(len(codes) // 7):
        f = codes[7 * i:7 * i + 7]
        l1.append(f[0])
        l2.append(f[1] - cb)
        l3.append(f[2] - 2 * cb)
        l3.append(f[3] - 3 * cb)
        l2.append(f[4] - 4 * cb)
        l3.append(f[5] - 5 * cb)
        l3.append(f[6] - 6 * cb)
    dev = loaded["device"]
    # Clamp to valid codebook range so a stray token can't crash the SNAC decoder.
    layers = [torch.tensor(l, dtype=torch.int64, device=dev).clamp_(0, cb - 1).unsqueeze(0)
              for l in (l1, l2, l3)]
    with torch.no_grad():
        audio = loaded["snac"].decode(layers)
    return _to_numpy(audio)


# ---- synthesis -------------------------------------------------------------

def synthesize(req: SpeechRequest) -> tuple[bytes, str, float]:
    st = get_status()
    loaded = _state.get("loaded")
    if loaded is None or st["state"] != "ready":
        detail = (f"model is loading ({st['model']}); retry shortly"
                  if st["state"] == "loading"
                  else (st.get("error") or "model not loaded yet"))
        raise HTTPException(503, detail)
    if not req.input.strip():
        raise HTTPException(400, "input is empty")

    req.voice = _resolve_voice_name(MODEL_KEY, req.voice)
    # Pin the RNG so a voice is reproducible across requests; every sampler otherwise
    # picks a different voice each call. Kokoro is deterministic and unaffected.
    seed = req.seed if req.seed is not None else _DEFAULT_SEED
    if seed is not None:
        import torch
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
    backend = loaded["backend"]
    sr = loaded["sr"]
    lang = (req.language or "").strip().lower()
    lang_code = lang if lang and lang != "unknown" else "en"

    if backend == "kokoro":
        voice_label = req.voice if _is_kokoro_voice(req.voice) else "af_heart"
        pipe = _kokoro_pipeline(voice_label[0])
        chunks = [_to_numpy(audio) for _, _, audio in
                  pipe(req.input, voice=voice_label, speed=req.speed)]
        if not chunks:
            raise HTTPException(500, "no audio produced")
        samples = np.concatenate(chunks)
    elif backend == "orpheus":
        voice_label = req.voice if _is_orpheus_voice(req.voice) else "tara"
        temp = req.temperature if req.temperature is not None else 0.6
        codes = _orpheus_generate_codes(loaded, req.input, voice_label, temp)
        if len(codes) < 7:
            raise HTTPException(500, "no audio produced")
        samples = _orpheus_decode(loaded, codes)
    elif backend == "higgs":
        ref = resolve_voice(req.voice)
        ref_text = (req.ref_text or resolve_ref_text(req.voice)) if ref else None
        voice_label = req.voice if (ref and ref_text) else "default"
        samples, sr = _higgs_generate(loaded, req.input, ref, ref_text, req.temperature)
    else:  # chatterbox
        ref = resolve_voice(req.voice)
        voice_label = req.voice if ref else "default"
        kwargs: dict = {}
        if ref:
            kwargs["audio_prompt_path"] = ref
        if req.exaggeration is not None:
            kwargs["exaggeration"] = req.exaggeration
        if req.cfg_weight is not None:
            kwargs["cfg_weight"] = req.cfg_weight
        wav = loaded["model"].generate(req.input, **kwargs)
        samples = _to_numpy(wav)

    preview = req.input.strip().replace("\n", " ")
    if len(preview) > 60:
        preview = preview[:57] + "..."
    print(f"[speech] backend={backend} voice={voice_label} fmt={req.response_format} "
          f"lang={lang_code} chars={len(req.input)} text={preview!r}")

    duration = len(samples) / sr
    audio, content_type = encode_audio(samples, sr, req.response_format)
    return audio, content_type, duration


def _serve(req: SpeechRequest) -> tuple[bytes, str, float]:
    """Worker-thread entry: load the active model if it was unloaded, then render."""
    if "loaded" not in _state:
        _load_model_on_worker(MODEL_KEY)
        if "loaded" not in _state:
            raise HTTPException(503, get_status().get("error") or "model failed to load")
    return synthesize(req)


def _check_auth(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid api key")


# ---- routes ----------------------------------------------------------------

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, authorization: str | None = Header(default=None)):
    import asyncio
    _check_auth(authorization)
    st = get_status()
    if st["state"] == "loading":
        raise HTTPException(503, f"model is loading ({st['model']}); retry shortly")
    _touch()
    loop = asyncio.get_running_loop()
    audio, content_type, duration = await loop.run_in_executor(_worker, _serve, req)
    _touch()
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
        {"id": m["key"], "label": m["label"], "backend": m["backend"],
         "object": "model", "owned_by": "local", "active": st["model"] == m["key"]}
        for m in MODEL_CATALOG
    ]}


@app.post("/v1/models/load")
async def load_model_route(req: LoadModelRequest, authorization: str | None = Header(default=None)):
    import asyncio
    _check_auth(authorization)
    key = (req.model or "").strip().lower()
    if key not in KNOWN_KEYS:
        raise HTTPException(400, f"unknown model '{req.model}'; choose from {sorted(KNOWN_KEYS)}")
    st = get_status()
    if st["state"] == "loading":
        raise HTTPException(409, f"already loading {st['model']}; retry when ready")
    if st["state"] == "ready" and st["model"] == key:
        return {"state": "ready", "model": key, "changed": False}
    _set_status(state="loading", model=key, error=None)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_worker, _load_model_on_worker, key)
    return {"state": "loading", "model": key, "changed": True}


@app.post("/v1/models/unload")
async def unload_model_route(authorization: str | None = Header(default=None)):
    """Drop the model now and free GPU/CPU memory. It reloads on the next request."""
    import asyncio
    _check_auth(authorization)
    if "loaded" not in _state:
        return {"state": get_status()["state"], "unloaded": False}
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_worker, _unload_on_worker)
    return {"state": "idle", "model": MODEL_KEY, "unloaded": True}


@app.get("/v1/audio/voices")
def voices(model: str | None = None):
    """Voices for a model. Defaults to the active model; pass ?model=<key> to
    preview another backend's voices without switching."""
    key = (model or get_status()["model"] or "chatterbox").strip().lower()
    if key not in KNOWN_KEYS:
        key = "chatterbox"
    return {"model": key, **_voices_for(key)}


@app.get("/health")
def health():
    st = get_status()
    return {"status": "ok", "state": st["state"], "model": st["model"],
            "loaded": st["state"] == "ready" and "loaded" in _state,
            "error": st["error"]}


@app.get("/ready")
def ready():
    """200 when the server can serve — either a model is in memory ("ready") or it
    is unloaded but loads on demand ("idle"). Use for the k8s readiness probe."""
    st = get_status()
    if st["state"] in ("ready", "idle"):
        return {"status": "ready", "model": st["model"],
                "loaded": st["state"] == "ready"}
    raise HTTPException(503, f"not ready (state={st['state']})")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("TTS_HOST", "0.0.0.0"),
                port=int(os.environ.get("TTS_PORT", "8000")))
