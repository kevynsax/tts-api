# Chatterbox TTS menu-bar app

A tiny macOS menu-bar app that runs the native MLX TTS server (`../server.py`)
and lets you start/stop it from the menu bar. It binds `0.0.0.0:8000` so the
k8s reverse proxy (`chatterbox-tts.kevyn.com.br` → this machine) can reach it.

## Build

```bash
./build.sh
```

Produces `ChatterboxTTS.app`.

## Install + run at login

```bash
cp -r ChatterboxTTS.app /Applications/
open /Applications/ChatterboxTTS.app
```

Then click the menu-bar icon and enable **Launch at Login**. (This registers the
app via `SMAppService`; it also appears under System Settings → General →
Login Items.)

## Menu

- **Running / Starting… / Stopped** — live status (polls `/health`).
- **Start / Stop / Restart** — control the server process.
- **Open Web UI** — opens `http://127.0.0.1:8000/docs`.
- **Launch at Login** — toggle auto-start.
- **Quit** — stops the server and exits.

## Notes

- The app spawns `../.venv-mlx/bin/python ../server.py`, so the `.venv-mlx`
  environment must exist (see the main tts-2 setup).
- The server takes a few seconds to load the model on start; the icon shows
  "Starting…" until `/health` reports ready.
- For "Launch at Login" to be stable, keep the app in a fixed location like
  `/Applications`.
