#!/usr/bin/env python3
"""Reference-image generation and editing through ComfyUI.

Two model families, both Apache 2.0 and both fitting a 16GB card:

  qwen-edit   Qwen-Image-Edit-2511 as a Q4_K_M GGUF with the 4-step Lightning
              LoRA. Takes one to three reference images and an instruction.
              With --angle it also loads fal's Multiple Angles LoRA and turns
              one render of an object into the same object from another
              camera position, which is what a 2.5D sprite needs for its top,
              front and three-quarter views.
  klein       FLUX.2 klein 4B (distilled, 4 steps). Text-to-image when no
              reference is given, reference-conditioned editing otherwise.
              Fast enough for iteration and takes LoRAs.

Angle vocabulary for --angle (fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA):
  azimuth   front | front-right quarter | right side | back-right quarter |
            back | back-left quarter | left side | front-left quarter
  elevation low-angle shot (-30) | eye-level shot (0) | elevated shot (30) | high-angle shot (60)
  distance  close-up | medium shot | wide shot
so  --angle "right side view high-angle shot medium shot"

    edit_asset.py --model qwen-edit --ref hull.png --prompt "..." --output out.png
    edit_asset.py --model qwen-edit --ref hull.png --angle "front view high-angle shot medium shot" --output top.png
    edit_asset.py --model klein --prompt "..." --output out.png
    edit_asset.py --model klein --ref hull.png --prompt "same ship, top view" --output out.png
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_asset as ga
from remove_background import upload_image

QWEN = {
    "unet": "qwen-image-edit-2511-Q4_K_M.gguf",
    "clip": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": "qwen_image_vae.safetensors",
    "lightning": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "angles": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
}
KLEIN = {
    "unet": "flux-2-klein-4b.safetensors",
    "clip": "qwen_3_4b.safetensors",
    "vae": "flux2-vae.safetensors",
}


def qwen_workflow(prompt, refs, seed, angle=None, angle_strength=0.9, lightning=True,
                  steps=None, cfg=None, negative=""):
    wf = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": QWEN["unet"]}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN["clip"], "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN["vae"]}},
    }
    model = ["1", 0]
    if lightning:
        wf["10"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model, "lora_name": QWEN["lightning"], "strength_model": 1.0}}
        model = ["10", 0]
    if angle:
        wf["11"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model, "lora_name": QWEN["angles"], "strength_model": angle_strength}}
        model = ["11", 0]
        prompt = f"<sks> {angle}" + (f", {prompt}" if prompt else "")
    wf["12"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": model, "shift": 3.1}}
    wf["13"] = {"class_type": "CFGNorm", "inputs": {"model": ["12", 0], "strength": 1.0}}

    images = {}
    for i, ref in enumerate(refs[:3], 1):
        wf[f"2{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf[f"3{i}"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": [f"2{i}", 0]}}
        images[f"image{i}"] = [f"3{i}", 0]

    wf["40"] = {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], **images}}
    wf["41"] = {"class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["2", 0], "prompt": negative, "vae": ["3", 0], **images}}
    wf["42"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["40", 0], "reference_latents_method": "index_timestep_zero"}}
    wf["43"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["41", 0], "reference_latents_method": "index_timestep_zero"}}
    # The output latent starts from the first reference, so the edit keeps its size.
    wf["44"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["31", 0], "vae": ["3", 0]}}
    wf["7"] = {"class_type": "KSampler", "inputs": {
        "model": ["13", 0], "positive": ["42", 0], "negative": ["43", 0], "latent_image": ["44", 0],
        "seed": seed, "steps": steps or (4 if lightning else 40), "cfg": cfg if cfg is not None else (1.0 if lightning else 3.0),
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "qwen_edit", "images": ["8", 0]}}
    return wf


def klein_workflow(prompt, refs, seed, width, height, steps=4, cfg=1.0, lora=None, lora_strength=1.0):
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN["clip"], "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
    }
    model = ["1", 0]
    if lora:
        wf["10"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model, "lora_name": lora, "strength_model": lora_strength}}
        model = ["10", 0]
    pos, neg = ["4", 0], ["5", 0]
    for i, ref in enumerate(refs, 1):
        wf[f"2{i}"] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf[f"3{i}"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [f"2{i}", 0], "upscale_method": "nearest-exact", "megapixels": 1.0, "resolution_steps": 1}}
        wf[f"4{i}"] = {"class_type": "VAEEncode", "inputs": {"pixels": [f"3{i}", 0], "vae": ["3", 0]}}
        wf[f"5{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [f"4{i}", 0]}}
        wf[f"6{i}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [f"4{i}", 0]}}
        pos, neg = [f"5{i}", 0], [f"6{i}", 0]
    wf["70"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    wf["71"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}}
    wf["72"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["73"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    wf["74"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": pos, "negative": neg, "cfg": cfg}}
    wf["75"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["73", 0], "guider": ["74", 0], "sampler": ["72", 0], "sigmas": ["71", 0], "latent_image": ["70", 0]}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["75", 0], "vae": ["3", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "klein", "images": ["8", 0]}}
    return wf


def main():
    p = argparse.ArgumentParser(description="Reference-image editing through ComfyUI.")
    p.add_argument("--model", choices=("qwen-edit", "klein"), required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--ref", action="append", default=[], help="reference image path (repeatable, up to 3)")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--angle", default=None, help="qwen-edit: Multiple Angles LoRA camera phrase")
    p.add_argument("--angle-strength", type=float, default=0.9)
    p.add_argument("--no-lightning", action="store_true", help="qwen-edit: 40 steps at cfg 3 instead of the 4-step LoRA")
    p.add_argument("--negative", default="")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cfg", type=float, default=None)
    p.add_argument("--width", type=int, default=1024, help="klein output width")
    p.add_argument("--height", type=int, default=1024, help="klein output height")
    p.add_argument("--lora", default=None, help="klein: LoRA filename from models/loras")
    p.add_argument("--lora-strength", type=float, default=1.0)
    a = p.parse_args()

    if a.model == "qwen-edit" and not a.ref:
        raise SystemExit("qwen-edit needs at least one --ref")
    seed = a.seed if a.seed is not None else random.randint(0, 2**32 - 1)
    ga.wait_for_server()
    refs = [upload_image(r) for r in a.ref]
    if a.model == "qwen-edit":
        wf = qwen_workflow(a.prompt, refs, seed, angle=a.angle, angle_strength=a.angle_strength,
                           lightning=not a.no_lightning, steps=a.steps, cfg=a.cfg, negative=a.negative)
    else:
        wf = klein_workflow(a.prompt, refs, seed, a.width, a.height, steps=a.steps or 4,
                            cfg=a.cfg if a.cfg is not None else 1.0, lora=a.lora, lora_strength=a.lora_strength)
    t = time.time()
    pid = ga.submit(wf)
    print(f"queued {pid} ({a.model}, seed {seed})", flush=True)
    size = ga.download(ga.first_image(ga.poll(pid)), a.output)
    print(f"wrote {a.output} ({size/1024:.0f} KiB) in {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()
