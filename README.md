# Chatterbox TTS

Local, OpenAI-compatible text-to-speech using [Chatterbox](https://github.com/resemble-ai/chatterbox),
running on Apple Silicon via MLX. Two ways to run it:

- **Web server** (`server.py`) — an HTTP service speaking OpenAI's audio API,
  served at `tts-macbook.kevyn.com.br` through a k8s reverse proxy.
- **Menu-bar app** (`menubar/`) — a macOS taskbar app that starts/stops the
  server and launches it at login.

It supports a built-in voice plus zero-shot voice cloning from short reference
clips in `voices/`.

---

## Install

Requires Python 3.11 (PyTorch/MLX have no 3.14 wheels) and `ffmpeg`.

```bash
brew install python@3.11 ffmpeg

cd ~/projects/tts-2
python3.11 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install -r requirements-mlx.txt
```

The model weights (~2 GB) download from Hugging Face into `~/.cache/huggingface`
on first run; after that it works offline.

---

## Web server

### Run

```bash
./run-server.sh                 # binds 127.0.0.1:8000, frees the port first
```

Bind on all interfaces (required when a reverse proxy or container reaches it):

```bash
TTS_HOST=0.0.0.0 ./run-server.sh
```

Environment:

| Var | Default | Meaning |
|-----|---------|---------|
| `TTS_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose) |
| `TTS_PORT` | `8000` | Port |
| `TTS_MODEL` | _(unset)_ | MLX model id; overrides `CHATTERBOX_MODEL`. Use `mlx-community/fish-audio-s2-pro-bf16` for OpenAudio/Fish |
| `CHATTERBOX_MODEL` | `mlx-community/chatterbox-fp16` | MLX model id (fallback) |
| `TTS_API_KEY` | _(unset)_ | If set, clients must send `Authorization: Bearer <key>` |

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/audio/speech` | Generate audio (OpenAI-compatible) |
| GET | `/v1/models` | List switchable models + which is active |
| POST | `/v1/models/load` | Switch the active model (hot-swap, no restart) |
| GET | `/v1/audio/voices` | List voices (built-in + clones) |
| GET | `/health` | Liveness + load state (`loading`/`ready`/`error`) |
| GET | `/openapi.json`, `/docs` | OpenAPI / Swagger (used as the availability probe) |

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello there.","voice":"default","language":"en","response_format":"mp3"}' \
  -o out.mp3
```

The server loads the model in the background, so it answers `/health` immediately
with `{"state":"loading"}` and flips to `"ready"` once the model is in memory.
Speech requests return `503` while loading. Switch models at runtime:

```bash
# model = catalog key ("chatterbox", "openaudio") or a full MLX repo id
curl -X POST http://127.0.0.1:8000/v1/models/load \
  -H "Content-Type: application/json" -d '{"model":"openaudio"}'
# then poll until ready
curl -s http://127.0.0.1:8000/health
```

A failed switch (e.g. bad repo) keeps the previously loaded model serving and
reports the reason in `/health`'s `error` field.

Request fields: `input`, `voice`, `language` (ISO 639-1, e.g. `pt`),
`response_format` (`mp3` default, `wav`, `flac`, `opus`, `aac`, `pcm`), `speed`,
optional `exaggeration` / `cfg_weight` / `temperature`, and `ref_text` (Fish
cloning transcript). The response carries an `X-Audio-Duration-Seconds` header.

### Models

Four backends are supported and selectable from the menu-bar **Model** submenu,
the `POST /v1/models/load` endpoint, or `TTS_MODEL`:

- **Chatterbox** (`mlx-community/chatterbox-fp16`) — default; multilingual, clones
  from a reference clip alone.
- **OpenAudio / Fish S2 Pro** (`mlx-community/fish-audio-s2-pro-bf16`) — cloning
  requires the reference clip's transcript (see below).
- **Kokoro** (`mlx-community/Kokoro-82M-bf16`) — small/fast; no cloning. Uses named
  voices (e.g. `af_heart` default, `am_michael`, `bf_emma`) passed in the `voice`
  field; `language` is mapped to Kokoro's codes (en→a, en-gb→b, pt→p, es→e, fr→f,
  hi→h, it→i, ja→j, zh→z). Needs the `misaki` G2P dependency (in `requirements-mlx.txt`).
- **Orpheus** (`mlx-community/orpheus-3b-0.1-ft-4bit`) — English; eight named
  voices (`tara` default, plus `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`)
  passed in the `voice` field. Optional cloning from a reference clip + its
  transcript (like Fish). Honors `temperature`; ignores `language`/`speed`.
  Supports inline emotion tags such as `<laugh>`, `<sigh>`, `<gasp>` in the text.

### Voices / cloning

Drop a reference clip at `voices/<name>.mp3` (or `.wav/.flac/.m4a`) and request
`"voice": "<name>"`. `"default"` or unknown names use the built-in voice.

`GET /v1/audio/voices` is per-model: it returns the voices the active model
accepts (Kokoro's 54 named voices, Orpheus's 8 named voices + your clone clips,
or `default` + clone clips for Chatterbox/Fish) plus a `cloning` flag. Pass
`?model=<key|repo>` to preview another backend's voices without switching.

**OpenAudio/Fish and Orpheus cloning** additionally need the transcript of the
reference clip. Provide it either per-request via the `ref_text` field, or as a
sidecar file `voices/<name>.txt` containing exactly what the clip says. Without a
transcript, Fish falls back to its built-in voice and Orpheus to a named voice
(both log a note). Chatterbox ignores `ref_text` and clones from the clip alone.

---

## Menu-bar app (taskbar)

A native macOS menu-bar app that runs the MLX server and starts it at login. It
binds `0.0.0.0:8000` so the reverse proxy can reach this machine.

### Build & install

```bash
cd menubar
./build.sh
cp -r ChatterboxTTS.app /Applications/
open /Applications/ChatterboxTTS.app
```

Click the menu-bar icon and enable **Launch at Login** (registers via
`SMAppService`; also visible under System Settings → General → Login Items).
Keep the app in `/Applications` so the login item path stays stable.

### Controls

- **Running / Starting… / Stopped** — live status (polls `/health`).
- **Start / Stop / Restart** — control the server.
- **Open Web UI** — opens `/docs`.
- **Quit** — stops the server and exits.

See `menubar/README.md` for details.

---

## Deploying the web version

The MLX server is fastest natively on this machine and **cannot run in a Linux
container** (MLX needs Apple Silicon + Metal). The production setup runs the
native server on the laptop and exposes it through the cluster:

1. Run the server on the laptop bound to all interfaces — easiest via the
   menu-bar app (which sets `TTS_HOST=0.0.0.0`), or `TTS_HOST=0.0.0.0 ./run-server.sh`.
2. The k8s manifests in `WebstormProjects/k8s/ai-features` reverse-proxy
   `tts-macbook.kevyn.com.br` to this machine (`192.168.10.179:8000`) via a
   selector-less Service + Endpoints. The laptop is often offline; consumers
   probe `/openapi.json` to detect that.

### Container / Kubernetes (x86_64, GPU)

`Dockerfile` builds a PyTorch variant (`server_cpu.py`) with the same HTTP API,
serving the **Chatterbox**, **Kokoro**, **Orpheus**, and **Higgs Audio v2** backends
(runtime-switchable via `/v1/models/load`) plus a `/ready` endpoint for the k8s
readiness probe. It defaults to CUDA torch (GPU) and is deployed in-cluster at
`tts.kevyn.com.br`. OpenAudio/Fish is MLX-host-only and not in this image.

Higgs Audio v2 on the container is the `bosonai/higgs-audio-v2-generation-3B-base`
model + its audio tokenizer. The bf16 weights want ~24 GB VRAM, so the server loads
it **4-bit quantized** (bitsandbytes NF4, ~5 GB) by default — set `HIGGS_QUANT_BITS=8`
(~7 GB) or `0` (full bf16) to change. It supports optional voice cloning via a
reference clip + transcript (`voices/<name>.wav` + `voices/<name>.txt`, or `ref_text`).

Orpheus on the container is a 3B Llama LM (defaults to the ungated
`unsloth/orpheus-3b-0.1-ft` mirror so it loads with no HF token; override with
`ORPHEUS_REPO`, e.g. the gated `canopylabs/orpheus-3b-0.1-ft` + an `HF_TOKEN`) plus
the SNAC codec; it needs a GPU node and ~6 GB of weights. It uses the eight named
voices (no cloning here); `temperature` is honored.

The deployment manifests live in `WebstormProjects/k8s/ai-features`
(`tts-server.yaml` + the shared `ingress.yaml`). Build & push the image:

```bash
# version in ./version.txt
container build --platform linux/amd64 \
  -t registry.kevyn.com.br/ai-features/tts-server:1.0.0 .
container push registry.kevyn.com.br/ai-features/tts-server:1.0.0
```

For a CPU-only image: `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu`.

**Build caching** (avoid reinstalling the multi-GB deps every build):

- `Dockerfile` uses BuildKit `--mount=type=cache` for pip/apt, so wheels aren't
  re-downloaded across builds; a persistent builder also reuses the whole layer.
- For ephemeral CI runners, build the deps once into a base image with
  `Dockerfile.base` (push as `tts-base`), then build per-commit with
  `Dockerfile.app` (`--build-arg BASE_IMAGE=…/tts-base:1.0.0`) — code-only,
  builds in seconds.

---

## CLI

Quick one-off synthesis without the server:

```bash
source .venv-mlx/bin/activate
python tts_mlx.py "Hello, this is Chatterbox on MLX."
python tts_mlx.py "My cloned voice." --voice samples/me.wav -o cloned
```

A PyTorch/CPU variant (`tts.py`, `.venv`) also exists but is ~7x slower than MLX.

---

## Notes

- MLX runs ~2.6x realtime on M1 Max (~3.3 GB); the PyTorch/MPS path is ~0.37x.
- Chatterbox embeds an inaudible Perth watermark in generated audio.
- MLX GPU streams are thread-local, so the server pins all model work to one
  worker thread; requests are serialized (correct for a single-user reader).
