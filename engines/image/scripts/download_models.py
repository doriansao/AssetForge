#!/usr/bin/env python3
"""Fetch image model weights into the ComfyUI model tree.

Files already present at the expected size are left alone, so this is safe to
re-run and safe to interrupt.

Sources are chosen to be ungated. black-forest-labs/FLUX.1-schnell requires
Hugging Face authentication, so the Flux UNET and VAE come from equivalent
public mirrors and are renamed to the names the workflows expect.

  python download_models.py              # the default model only
  python download_models.py --all        # every model in the registry
  python download_models.py --model sdxl --loras
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths

# Stage downloads inside the repo: the default cache lives on the system drive,
# which has no room for tens of GB of weights.
os.environ.setdefault("HF_HOME", str(paths.CACHE))

from huggingface_hub import hf_hub_download

import models as registry

# model key -> list of (repo, remote filename, target subdir, local name, bytes)
DOWNLOADS = {
    "flux-schnell": [
        ("Kijai/flux-fp8", "flux1-schnell-fp8-e4m3fn.safetensors",
         "unet", "flux1-schnell-fp8.safetensors", 11891329784),
        ("StableDiffusionVN/Flux", "Vae/flux_vae.safetensors",
         "vae", "ae.safetensors", 335304388),
        ("comfyanonymous/flux_text_encoders", "clip_l.safetensors",
         "clip", "clip_l.safetensors", 246144152),
        ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors",
         "clip", "t5xxl_fp8_e4m3fn.safetensors", 4893934904),
    ],
    "sdxl": [
        ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors",
         "checkpoints", "sd_xl_base_1.0.safetensors", 6938078334),
    ],
    "sd15": [
        ("stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors",
         "checkpoints", "v1-5-pruned-emaonly.safetensors", 4265146304),
    ],
}

LORAS = [
    ("nerijs/pixel-art-xl", "pixel-art-xl.safetensors",
     "loras", "pixel-art-xl.safetensors", 170543052),
    ("Muapi/top-down-pixel-art-flux-lora", "top-down-pixel-art-flux-lora.safetensors",
     "loras", "top-down-pixel-art-flux-lora.safetensors", 77161056),
]


def fetch(repo: str, remote: str, subdir: str, local: str, expected: int) -> None:
    dest = paths.MODELS / subdir / local
    if dest.exists() and dest.stat().st_size == expected:
        print(f"[skip] {local} ({expected / 2**30:.2f} GiB)", flush=True)
        return

    print(f"[get ] {repo}/{remote}", flush=True)
    cached = hf_hub_download(repo_id=repo, filename=remote)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)

    size = dest.stat().st_size
    if size != expected:
        print(f"[WARN] {local}: got {size} bytes, expected {expected}", flush=True)
    print(f"[done] {local} ({size / 2**30:.2f} GiB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", choices=sorted(DOWNLOADS),
                    help="repeatable; default is just the engine default model")
    ap.add_argument("--all", action="store_true", help="every model in the registry")
    ap.add_argument("--loras", action="store_true", help="also fetch the pixel-art LoRAs")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, spec in registry.MODELS.items():
            total = sum(j[4] for j in DOWNLOADS.get(name, []))
            print(f"{name:14s} {total / 2**30:6.2f} GiB  {spec['notes']}")
        print(f"{'loras':14s} {sum(j[4] for j in LORAS) / 2**30:6.2f} GiB")
        return 0

    wanted = sorted(DOWNLOADS) if args.all else (args.model or [registry.DEFAULT_MODEL])
    jobs = [j for name in wanted for j in DOWNLOADS[name]]
    if args.loras or args.all:
        jobs += LORAS

    print(f"{len(jobs)} files, {sum(j[4] for j in jobs) / 2**30:.1f} GiB total\n")
    for job in jobs:
        fetch(*job)
    print(f"\nmodels in {paths.MODELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
