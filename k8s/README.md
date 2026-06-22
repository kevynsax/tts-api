# TTS Server on Kubernetes (x86_64)

Runs the PyTorch server (`server_cpu.py`) — **Chatterbox** + **Kokoro** backends,
CPU or CUDA. (The native MLX `server.py` / OpenAudio are Apple-Silicon-only and
are not part of this image.)

Same HTTP API as the Mac server: `POST /v1/audio/speech`, `GET /v1/models`,
`POST /v1/models/load`, `GET /v1/audio/voices`, `GET /health`, `GET /ready`.

## 1. Build & push (on your x86_64 builder)

```bash
# from the repo root (the Dockerfile is there)
REG=your-registry.example.com/yourns
docker build --platform linux/amd64 -t $REG/tts-server:latest .
docker push $REG/tts-server:latest
```

Cross-building from another arch instead? Use buildx:

```bash
docker buildx build --platform linux/amd64 -t $REG/tts-server:latest --push .
```

## 2. Deploy

Set your image in `deployment.yaml` (replace `REGISTRY/tts-server:latest`), then:

```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml      # optional
```

Or in one shot (after editing the image):

```bash
kubectl apply -f k8s/
```

## 3. Verify

```bash
kubectl rollout status deploy/tts-server
kubectl port-forward svc/tts-server 8000:80

curl localhost:8000/health      # {"state":"loading"|"ready", ...}
curl localhost:8000/v1/models   # chatterbox + kokoro, which is active
curl -X POST localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Kubernetes.","response_format":"mp3"}' -o out.mp3
```

Switch model at runtime (no restart):

```bash
curl -X POST localhost:8000/v1/models/load \
  -H 'Content-Type: application/json' -d '{"model":"kokoro"}'
```

## Caching the build (avoid reinstalling deps every time)

The slow part is the multi-GB `pip install` (torch + chatterbox-tts + kokoro).
Three levels, cheapest first:

1. **Layer cache (automatic).** On a builder that keeps its cache, rebuilding the
   same `Dockerfile` reuses the whole dependency layer — pip doesn't run again
   unless that `RUN` line (or something above it) changes.
2. **pip wheel cache (built in).** `Dockerfile` mounts `type=cache` on pip's cache
   dir, so even when the layer is invalidated the wheels aren't re-downloaded.
3. **Base image (best for ephemeral CI runners).** CI runners usually start with an
   empty cache, defeating 1–2. Instead, bake the deps once into a base image and
   have per-commit builds layer only the code on top:

   ```bash
   # once (and only when dependencies change):
   container build --platform linux/amd64 \
     -t registry.kevyn.com.br/ai-features/tts-base:1.0.0 -f Dockerfile.base .
   container push registry.kevyn.com.br/ai-features/tts-base:1.0.0

   # every commit (fast — no pip install):
   container build --platform linux/amd64 \
     --build-arg BASE_IMAGE=registry.kevyn.com.br/ai-features/tts-base:1.0.0 \
     -t registry.kevyn.com.br/ai-features/tts-server:1.0.0 -f Dockerfile.app .
   container push registry.kevyn.com.br/ai-features/tts-server:1.0.0
   ```

   `Dockerfile.base` and `Dockerfile.app` are at the repo root. (`Dockerfile`
   remains a self-contained build that needs no base image.)

## Notes

- **First start is slow**: weights (~2GB Chatterbox / ~0.4GB Kokoro) download into
  the PVC-backed HF cache. The `startupProbe` allows ~10 min; subsequent restarts
  reuse the cache. `/ready` returns 200 only once the model is in memory, so no
  traffic is routed before then.
- **Memory**: needs ~6–8GB to load the torch model; the deployment requests 6Gi /
  limits 10Gi. Bump if you hit OOMKilled.
- **GPU**: on a CUDA node, set `TTS_DEVICE=cuda` and add `nvidia.com/gpu: "1"` to
  the container `resources.limits` (commented in `deployment.yaml`).
- **Voices / cloning**: bundled `voices/*.mp3` are baked into the image; request
  `"voice":"<name>"` (Chatterbox). Kokoro uses named voices (e.g. `af_heart`,
  `am_michael`) and ignores cloning.
- **Auth**: set `TTS_API_KEY` (env or Secret) to require `Authorization: Bearer …`.
