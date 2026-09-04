#!/usr/bin/env python3
"""Registry of image models the engine can drive.

Each entry pins the ComfyUI API workflow, the node ids the generator rewrites,
and sampler defaults appropriate to that model. Node ids are deliberately kept
consistent across workflows -- 4 is the positive prompt, 5 the negative, 6 the
latent, 7 the sampler -- so the calling code stays model-agnostic. Only the
loader section differs, which is what `model_out` and `clip_out` describe.

Why more than one model: they fail in different directions, and which one is
right depends on the asset.

  flux-schnell  Best prompt adherence of the three, and fast at 4 steps. But it
                is a diffusion transformer, so it has no convolution layers,
                which rules out circular-padding seamless generation, and its
                LoRA ecosystem is thin and trained for FLUX.1-dev rather than
                schnell (measured unreliable -- see docs/MODELS.md).
  sdxl          Slower and needs more prompt care, but it is a UNet with by far
                the largest pixel-art LoRA ecosystem. pixel-art-xl is the
                reference pixel-art LoRA and only exists for SDXL.
  sd15          Oldest and weakest at prompt following, but tiny, very fast, and
                has the deepest ControlNet ecosystem, which is what you want for
                pose-controlled character sprite sheets.
"""
from __future__ import annotations

MODELS = {
    "flux-schnell": {
        "workflow": "flux_schnell.json",
        "model_out": ["1", 0],       # UNETLoader
        "clip_out": ["2", 0],        # DualCLIPLoader
        "defaults": {
            "width": 1024, "height": 1024, "steps": 4, "cfg": 1.0,
            "sampler": "euler", "scheduler": "simple",
        },
        "supports_negative": False,  # guidance-distilled, cfg is pinned at 1.0
        "files": [
            ("unet", "flux1-schnell-fp8.safetensors"),
            ("vae", "ae.safetensors"),
            ("clip", "clip_l.safetensors"),
            ("clip", "t5xxl_fp8_e4m3fn.safetensors"),
        ],
        "notes": "Default. Best prompt adherence, ~5s per 1024px render.",
    },
    "sdxl": {
        "workflow": "sdxl.json",
        "model_out": ["1", 0],       # CheckpointLoaderSimple
        "clip_out": ["1", 1],
        "defaults": {
            "width": 1024, "height": 1024, "steps": 25, "cfg": 7.0,
            "sampler": "dpmpp_2m", "scheduler": "karras",
        },
        "supports_negative": True,
        "files": [("checkpoints", "sd_xl_base_1.0.safetensors")],
        "notes": "Use with the pixel-art-xl LoRA. Slower, but the LoRA and "
                 "ControlNet ecosystem is far deeper than Flux's.",
    },
    "sd15": {
        "workflow": "sd15.json",
        "model_out": ["1", 0],
        "clip_out": ["1", 1],
        "defaults": {
            "width": 512, "height": 512, "steps": 25, "cfg": 7.0,
            "sampler": "dpmpp_2m", "scheduler": "karras",
        },
        "supports_negative": True,
        "files": [("checkpoints", "v1-5-pruned-emaonly.safetensors")],
        "notes": "Fast and small. Weakest prompt adherence; best ControlNet support.",
    },
}

DEFAULT_MODEL = "flux-schnell"

# Negative prompts only matter for the cfg-guided models. Kept here so every
# entry point uses the same baseline rather than inventing its own.
DEFAULT_NEGATIVE = ("blurry, jpeg artifacts, watermark, signature, text, "
                    "photorealistic, 3d render, smooth gradients, anti-aliased")


def get(name: str) -> dict:
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; available: {', '.join(MODELS)}")
    return MODELS[name]
