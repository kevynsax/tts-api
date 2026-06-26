# Project rules

## Both deployments must stay in sync

This project ships two deployments of the same TTS API:

- `server.py` — MLX backend (Apple Silicon, native).
- `server_cpu.py` — PyTorch/CPU backend (mirrors the same HTTP API).

**Every feature or change that is applicable to both must be implemented in both.**
When a feature is requested, implement it in `server.py` *and* `server_cpu.py` in the
same change. Keep shared data (voice-name aliases, model catalog, request schema,
audio encoding, endpoints) consistent across the two files.

Only skip one side when the feature is genuinely impossible there — e.g. a backend
specific to one runtime (MLX-only models like OpenAudio/Fish, or MLX-only fixes like
the bf16 Snake fp32 patch). When you skip a side, say so explicitly and why.

After changing either file, verify both still import / parse.
