#!/usr/bin/env python
"""
Chatterbox TTS CLI for Apple Silicon (M1 Max).

Examples:
    # Default voice
    python tts.py "Hello, this is Chatterbox running locally on my Mac."

    # Clone a voice from a 5-15s reference clip
    python tts.py "Cloned voice speaking." --voice samples/me.wav -o cloned.wav

    # More emotional / expressive delivery
    python tts.py "I can't believe it!" --exaggeration 0.8 --cfg 0.3
"""
import argparse
import time

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def patch_torch_load(device: str) -> None:
    """Chatterbox checkpoints are saved as CUDA tensors. On Mac/CPU we must
    remap them at load time or `from_pretrained` will fail."""
    if device == "cuda":
        return
    map_location = torch.device(device)
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("map_location", map_location)
        return original(*args, **kwargs)

    torch.load = patched


def main() -> None:
    p = argparse.ArgumentParser(description="Chatterbox TTS (local, Apple Silicon).")
    p.add_argument("text", help="Text to synthesize.")
    p.add_argument("-o", "--output", default="output.wav", help="Output WAV path.")
    p.add_argument("--voice", default=None,
                   help="Reference audio (wav/mp3, ~5-15s) to clone. Omit for default voice.")
    p.add_argument("--exaggeration", type=float, default=0.5,
                   help="Emotion intensity 0.25-1.0 (default 0.5).")
    p.add_argument("--cfg", type=float, default=0.5,
                   help="CFG/pace weight 0.0-1.0; lower = slower, more deliberate (default 0.5).")
    args = p.parse_args()

    device = pick_device()
    patch_torch_load(device)

    print(f"[chatterbox] loading model on {device.upper()} (first run downloads ~2GB)...")
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"[chatterbox] model ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    wav = model.generate(
        args.text,
        audio_prompt_path=args.voice,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg,
    )
    gen = time.time() - t0
    audio_sec = wav.shape[-1] / model.sr
    ta.save(args.output, wav, model.sr)
    print(f"[chatterbox] wrote {args.output} "
          f"({audio_sec:.1f}s audio in {gen:.1f}s -> {audio_sec / gen:.1f}x realtime)")


if __name__ == "__main__":
    main()
