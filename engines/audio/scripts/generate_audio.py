#!/usr/bin/env python3
"""Generate game music and sound effects locally with AudioCraft.

Must run under the audio engine's own virtualenv, not the system Python:
AudioCraft pins an older torch than the image engine uses, which is why the two
engines have separate environments.

  engines/audio/venv/Scripts/python.exe scripts/generate_audio.py --kind sfx \
      --prompt "heavy wooden door slamming shut" --output out/audio/door.wav

Generation is the easy half. Raw output starts with silence, ends mid-phrase and
sits at an arbitrary level, so everything goes through audio_processor before it
is written. Pass --raw to also keep the untouched version.

Models are downloaded on first use and cached under .cache/ at the repo root.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths

os.environ.setdefault("HF_HOME", str(paths.CACHE))
os.environ.setdefault("AUDIOCRAFT_CACHE_DIR", str(paths.CACHE / "audiocraft"))

import numpy as np
import soundfile as sf
import torch

import audio_processor as post
import models as registry

_LOADED: dict[str, object] = {}


def load(name: str):
    """Load and cache a model. Only one is worth holding on a 16GB card."""
    if name in _LOADED:
        return _LOADED[name]

    spec = registry.get(name)
    from audiocraft.models import AudioGen, MusicGen

    cls = MusicGen if spec["kind"] == "music" else AudioGen
    print(f"loading {name} ({spec['params']}, {spec['vram']})...", flush=True)
    started = time.time()
    model = cls.get_pretrained(spec["repo"])
    print(f"loaded in {time.time()-started:.1f}s", flush=True)

    _LOADED.clear()          # drop the previous model before keeping this one
    _LOADED[name] = model
    return model


def generate(prompt: str, model_name: str, duration: float, count: int = 1,
             seed: int | None = None, temperature: float = 1.0,
             top_k: int = 250, cfg: float = 3.0):
    """Return (list of float32 arrays, sample rate)."""
    model = load(model_name)
    if seed is not None:
        torch.manual_seed(seed)

    model.set_generation_params(duration=duration, temperature=temperature,
                                top_k=top_k, cfg_coef=cfg)
    with torch.no_grad():
        wav = model.generate([prompt] * count, progress=True)

    # audiocraft returns [batch, channels, samples] on the GPU.
    out = [w.detach().cpu().numpy().T.squeeze() for w in wav]
    return out, model.sample_rate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--kind", choices=("music", "sfx"), default="sfx")
    ap.add_argument("--model", default=None, help="override the default for --kind")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds; default 3 for sfx, 15 for music")
    ap.add_argument("--count", type=int, default=1, help="generate N variations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=3.0, help="prompt adherence")
    ap.add_argument("--loop", action="store_true", help="force seamless looping")
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--crossfade-ms", type=float, default=400.0)
    ap.add_argument("--raw", action="store_true", help="also keep the unprocessed file")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, spec in registry.MODELS.items():
            print(f"{name:18s} {spec['kind']:5s} {spec['params']:>5s} "
                  f"{spec['vram']:>8s}  {spec['sample_rate']}Hz")
            print(f"{'':18s} {spec['notes']}")
        return 0

    if not (args.prompt and args.output):
        raise SystemExit("--prompt and --output are required (or use --list)")

    model_name = args.model or registry.DEFAULTS[args.kind]
    duration = args.duration if args.duration is not None else (15.0 if args.kind == "music" else 3.0)
    loop = True if args.loop else (False if args.no_loop else None)

    started = time.time()
    clips, rate = generate(args.prompt, model_name, duration, args.count,
                           args.seed, args.temperature, cfg=args.cfg)

    paths.ensure(args.output.parent)
    for i, clip in enumerate(clips):
        target = args.output if args.count == 1 else \
            args.output.with_name(f"{args.output.stem}_{i+1:02d}{args.output.suffix}")

        if args.raw:
            raw_path = target.with_name(f"{target.stem}_raw.wav")
            sf.write(str(raw_path), clip, rate, subtype="PCM_16")

        processed, info = post.process(clip, rate, args.kind, loop=loop,
                                       crossfade_ms=args.crossfade_ms)
        post.save(processed, post.GAME_RATE, target)
        seam = f", seam {info['seam_ratio']:.2f}" if info["seam_ratio"] is not None else ""
        print(f"{target.name}  {info['out_seconds']:.1f}s @{info['out_rate']}Hz"
              f"{' looped' if info['looped'] else ''}{seam}")

    print(f"done in {time.time()-started:.1f}s ({model_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
