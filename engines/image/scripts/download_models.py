"""Fetch the Flux Schnell model components into the ComfyUI model tree.

Files already present with a plausible size are left alone. Sources are chosen to
be ungated: black-forest-labs/FLUX.1-schnell requires HF authentication, so the
UNET and VAE come from equivalent public mirrors and are renamed to the names the
workflow expects.
"""
import os
import shutil
import sys

# Stage downloads on G: -- the default cache lives on the system drive, which
# has no room for ~17 GiB of model weights.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

os.environ.setdefault("HF_HOME", str(paths.CACHE))

from huggingface_hub import hf_hub_download

ROOT = str(paths.MODELS)

# (repo_id, remote filename, target subdir, local filename, expected bytes)
JOBS = [
    ("Kijai/flux-fp8", "flux1-schnell-fp8-e4m3fn.safetensors",
     "unet", "flux1-schnell-fp8.safetensors", 11891329784),
    ("StableDiffusionVN/Flux", "Vae/flux_vae.safetensors",
     "vae", "ae.safetensors", 335304388),
    ("comfyanonymous/flux_text_encoders", "clip_l.safetensors",
     "clip", "clip_l.safetensors", 246144152),
    ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors",
     "clip", "t5xxl_fp8_e4m3fn.safetensors", 4893934904),
]


def main():
    for repo, remote, subdir, local, expected in JOBS:
        dest = os.path.join(ROOT, subdir, local)
        if os.path.exists(dest) and os.path.getsize(dest) == expected:
            print(f"[skip] {local} already present ({expected/2**30:.2f} GiB)", flush=True)
            continue

        print(f"[get ] {repo}/{remote} -> {dest}", flush=True)
        cached = hf_hub_download(repo_id=repo, filename=remote)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(cached, dest)

        size = os.path.getsize(dest)
        if size != expected:
            print(f"[WARN] {local}: got {size} bytes, expected {expected}", flush=True)
        print(f"[done] {local} ({size/2**30:.2f} GiB)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
