#!/usr/bin/env python
"""
Chatterbox TTS via MLX — fast native Apple Silicon inference (~2.6x realtime on M1 Max).

Use the .venv-mlx environment for this script:
    source .venv-mlx/bin/activate

Examples:
    # Default voice
    python tts_mlx.py "Hello, this is Chatterbox on MLX."

    # Voice cloning from a 5-15s reference clip (auto-transcribed)
    python tts_mlx.py "My cloned voice." --voice samples/me.wav -o cloned

    # Turbo build (better cloning / expressiveness)
    python tts_mlx.py "Expressive line!" --model mlx-community/chatterbox-turbo-fp16
"""
import argparse

from mlx_audio.tts.generate import generate_audio


def main() -> None:
    p = argparse.ArgumentParser(description="Chatterbox TTS via MLX (Apple Silicon).")
    p.add_argument("text", help="Text to synthesize.")
    p.add_argument("-o", "--output", default="mlx_out",
                   help="Output file prefix (e.g. 'cloned' -> cloned_000.wav).")
    p.add_argument("--model", default="mlx-community/chatterbox-fp16",
                   help="MLX model id. Options: chatterbox-fp16, chatterbox-turbo-fp16, "
                        "chatterbox-6bit, chatterbox-4bit.")
    p.add_argument("--voice", default=None,
                   help="Reference audio (wav/mp3) to clone. Omitted = default voice.")
    p.add_argument("--ref-text", default=None,
                   help="Transcript of the reference clip. If omitted, it's auto-transcribed.")
    p.add_argument("--cfg", type=float, default=None,
                   help="CFG scale; lower = slower/more deliberate.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default 0.7).")
    args = p.parse_args()

    generate_audio(
        text=args.text,
        model=args.model,
        ref_audio=args.voice,
        ref_text=args.ref_text,
        cfg_scale=args.cfg,
        temperature=args.temperature,
        file_prefix=args.output,
        audio_format="wav",
        verbose=True,
    )
    print(f"[mlx] done -> {args.output}_000.wav")


if __name__ == "__main__":
    main()
