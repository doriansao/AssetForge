#!/usr/bin/env python3
"""Render a single game asset through a running ComfyUI instance.

Loads the Flux Schnell API workflow, injects the prompt, dimensions and a random
seed, submits it to the ComfyUI REST API, waits for the render to finish and
writes the resulting PNG to the requested path.

With --pixel-art the render is additionally snapped onto a true pixel grid by
pixel_art_processor, and the full-resolution frame is kept alongside as
<output>_raw.png.

Only the standard library is used so this runs under the embedded ComfyUI
interpreter as well as a system Python.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import paths
import models as model_registry

SERVER = paths.SERVER

# Node ids, identical across every workflow in engines/image/workflows/.
NODE_POSITIVE = "4"
NODE_NEGATIVE = "5"
NODE_LATENT = "6"
NODE_SAMPLER = "7"
NODE_LORA = "10"  # added at runtime only when a LoRA is requested


def api(path, payload=None):
    """GET path, or POST it when payload is given. Returns decoded JSON."""
    url = f"{SERVER}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wait_for_server(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            api("/system_stats")
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"ComfyUI did not answer on {SERVER} within {timeout}s")


def build_workflow(prompt, width=None, height=None, seed=0, lora=None,
                   lora_strength=1.0, model=None, negative=None,
                   steps=None, cfg=None):
    """Load the workflow for `model` and rewrite it for this request.

    Node ids are the same across every workflow in workflows/ (4 positive,
    5 negative, 6 latent, 7 sampler), so only the loader wiring is model
    specific, and that comes from the registry.
    """
    model = model or model_registry.DEFAULT_MODEL
    spec = model_registry.get(model)
    defaults = spec["defaults"]

    with open(paths.WORKFLOWS / spec["workflow"], encoding="utf-8") as fh:
        wf = json.load(fh)

    wf[NODE_POSITIVE]["inputs"]["text"] = prompt
    wf[NODE_LATENT]["inputs"]["width"] = width or defaults["width"]
    wf[NODE_LATENT]["inputs"]["height"] = height or defaults["height"]

    sampler = wf[NODE_SAMPLER]["inputs"]
    sampler["seed"] = seed
    sampler["steps"] = steps if steps is not None else defaults["steps"]
    sampler["cfg"] = cfg if cfg is not None else defaults["cfg"]
    sampler["sampler_name"] = defaults["sampler"]
    sampler["scheduler"] = defaults["scheduler"]

    if spec["supports_negative"]:
        # Flux Schnell is guidance-distilled and runs at cfg 1.0, where the
        # negative branch has no effect, so it is left empty there.
        wf[NODE_NEGATIVE]["inputs"]["text"] = (
            negative if negative is not None else model_registry.DEFAULT_NEGATIVE)

    if lora:
        # Splice a LoraLoader between the loaders and their consumers. The LoRA
        # has to reach the text encoder as well as the denoiser: a style LoRA
        # shifts how its trigger word is embedded, not just how it is denoised.
        wf[NODE_LORA] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": spec["model_out"],
                "clip": spec["clip_out"],
                "lora_name": lora,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
            },
        }
        sampler["model"] = [NODE_LORA, 0]
        for node in wf.values():
            if node["class_type"] == "CLIPTextEncode":
                node["inputs"]["clip"] = [NODE_LORA, 1]
    return wf


def submit(wf):
    try:
        result = api("/prompt", {"prompt": wf, "client_id": f"gen-{random.randint(0, 1 << 30)}"})
    except urllib.error.HTTPError as exc:
        # ComfyUI returns the validation failure as a JSON body -- surface it.
        raise SystemExit(f"ComfyUI rejected the workflow:\n{exc.read().decode(errors='replace')}")
    return result["prompt_id"]


def poll(prompt_id, timeout=900):
    """Block until the prompt leaves the queue, then return its history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = api(f"/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                messages = json.dumps(status.get("messages", []), indent=2)
                raise SystemExit(f"Render failed:\n{messages}")
            if entry.get("outputs"):
                return entry
        time.sleep(1)
    raise SystemExit(f"Timed out after {timeout}s waiting for prompt {prompt_id}")


def first_image(entry):
    for node_output in entry["outputs"].values():
        for image in node_output.get("images", []):
            if image.get("type") != "temp":
                return image
    raise SystemExit("Render finished but produced no image output")


def download(image, output):
    query = urllib.parse.urlencode({
        "filename": image["filename"],
        "subfolder": image.get("subfolder", ""),
        "type": image.get("type", "output"),
    })
    with urllib.request.urlopen(f"{SERVER}/view?{query}", timeout=120) as resp:
        blob = resp.read()

    parent = os.path.dirname(os.path.abspath(output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output, "wb") as fh:
        fh.write(blob)
    return len(blob)


def main():
    parser = argparse.ArgumentParser(description="Generate a game asset with Flux Schnell.")
    parser.add_argument("--prompt", required=True, help="text prompt for the asset")
    parser.add_argument("--output", required=True, help="path to write the PNG to")
    parser.add_argument("--width", type=int, default=None, help="default: model native")
    parser.add_argument("--height", type=int, default=None, help="default: model native")
    parser.add_argument("--seed", type=int, default=None, help="omit for a random seed")

    pixel = parser.add_argument_group("pixel art post-processing")
    pixel.add_argument("--pixel-art", action="store_true",
                       help="snap the render onto a true pixel grid (see pixel_art_processor)")
    pixel.add_argument("--target-size", type=int, default=64,
                       help="longest side of the sprite in logical pixels (default 64)")
    pixel.add_argument("--palette-size", type=int, default=16,
                       help="k-means colour count (default 16; 32-48 for human characters)")
    pixel.add_argument("--upscale-factor", type=int, default=0,
                       help="also write a NEAREST-upscaled preview at this factor")
    pixel.add_argument("--keep-background", action="store_true",
                       help="skip background keying and keep the render opaque")
    style = parser.add_argument_group("style")
    style.add_argument("--lora", default=None,
                       help="LoRA filename from ComfyUI/models/loras (e.g. ume_modern_pixelart.safetensors)")
    style.add_argument("--lora-strength", type=float, default=1.0)
    style.add_argument("--model", default=model_registry.DEFAULT_MODEL,
                       choices=sorted(model_registry.MODELS),
                       help="which image model to drive (see docs/MODELS.md)")
    style.add_argument("--negative", default=None,
                       help="negative prompt; ignored by flux-schnell")
    style.add_argument("--steps", type=int, default=None, help="override sampler steps")
    style.add_argument("--cfg", type=float, default=None, help="override guidance scale")

    pixel.add_argument("--grid", choices=("detect", "force"), default="detect",
                       help="detect the native block size then resample (default), or "
                            "drive the grid walker at target-size cells directly")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)

    wait_for_server()
    wf = build_workflow(args.prompt, args.width, args.height, seed,
                        lora=args.lora, lora_strength=args.lora_strength,
                        model=args.model, negative=args.negative,
                        steps=args.steps, cfg=args.cfg)

    started = time.time()
    prompt_id = submit(wf)
    latent = wf[NODE_LATENT]["inputs"]
    print(f"queued {prompt_id} ({args.model}, seed {seed}, "
          f"{latent['width']}x{latent['height']})", flush=True)

    entry = poll(prompt_id)

    if not args.pixel_art:
        size = download(first_image(entry), args.output)
        print(f"wrote {args.output} ({size/1024:.0f} KiB) in {time.time()-started:.1f}s")
        return 0

    # Keep the full-resolution render: the snapper reads its grid from those
    # gradients, so the raw frame is the only thing that can be re-processed at
    # a different target size or palette later.
    stem, _ = os.path.splitext(args.output)
    raw_path = f"{stem}_raw.png"
    size = download(first_image(entry), raw_path)
    print(f"raw   {raw_path} ({size/1024:.0f} KiB)", flush=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from PIL import Image

    import pixel_art_processor as pap

    info = pap.process(Image.open(raw_path), target_size=args.target_size,
                       palette_size=args.palette_size,
                       remove_background=not args.keep_background, grid=args.grid)
    info["sprite"].save(args.output)
    print(f"sprite {args.output}  native {info['native_size'][0]}x{info['native_size'][1]}"
          f" -> {info['final_size'][0]}x{info['final_size'][1]}"
          f"  {info['colors']} colours  (block {info['detected_step']:.2f}px)")

    if args.upscale_factor:
        preview = pap.nearest_upscale(info["sprite"], scale=args.upscale_factor)
        preview_path = f"{stem}_x{args.upscale_factor}.png"
        preview.save(preview_path)
        print(f"preview {preview_path}  {preview.width}x{preview.height}")

    print(f"done in {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
