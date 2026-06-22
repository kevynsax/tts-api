# Chatterbox TTS

Local, OpenAI-compatible text-to-speech using [Chatterbox](https://github.com/resemble-ai/chatterbox),
running on Apple Silicon via MLX. Two ways to run it:

- **Web server** (`server.py`) — an HTTP service speaking OpenAI's audio API,
  served at `chatterbox-tts.kevyn.com.br` through a k8s reverse proxy.
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
| `CHATTERBOX_MODEL` | `mlx-community/chatterbox-fp16` | MLX model id |
| `TTS_API_KEY` | _(unset)_ | If set, clients must send `Authorization: Bearer <key>` |

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/audio/speech` | Generate audio (OpenAI-compatible) |
| GET | `/v1/audio/voices` | List voices (built-in + clones) |
| GET | `/health` | Liveness + model load state |
| GET | `/openapi.json`, `/docs` | OpenAPI / Swagger (used as the availability probe) |

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello there.","voice":"default","language":"en","response_format":"mp3"}' \
  -o out.mp3
```

Request fields: `input`, `voice`, `language` (ISO 639-1, e.g. `pt`),
`response_format` (`mp3` default, `wav`, `flac`, `opus`, `aac`, `pcm`), `speed`,
and optional `exaggeration` / `cfg_weight` / `temperature`. The response carries
an `X-Audio-Duration-Seconds` header.

### Voices / cloning

Drop a reference clip at `voices/<name>.mp3` (or `.wav/.flac/.m4a`) and request
`"voice": "<name>"`. `"default"` or unknown names use the built-in voice.

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
   `chatterbox-tts.kevyn.com.br` to this machine (`192.168.10.179:8000`) via a
   selector-less Service + Endpoints. The laptop is often offline; consumers
   probe `/openapi.json` to detect that.

### Container (portable, CPU only)

For environments without Apple Silicon, `Dockerfile` builds a PyTorch CPU
variant (`server_cpu.py`) with the same API. It's slower and needs ≥6 GB RAM.

```bash
container build -t chatterbox-tts .
container run -d --name chatterbox-tts --memory 6g --cpus 4 chatterbox-tts
```

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
