#!/usr/bin/env python3
"""Cut a rendered asset out of its background with BiRefNet, through ComfyUI.

BiRefNet is a matting model, so unlike the flood-fill keying in
pixel_art_processor it keeps near-white foreground (ivory hulls on a white
plate) and gives soft anti-aliased edges. Use it for any smooth, high
resolution sprite. Keep the flood-fill for true pixel art, where a soft edge is
wrong.

Uses ComfyUI's native LoadBackgroundRemovalModel / RemoveBackground nodes
(ComfyUI >= 0.3.60) and models/background_removal/birefnet.safetensors.

    remove_background.py --input render.png --output sprite.png [--crop] [--pad 8]
"""
import argparse
import io
import json
import os
import pathlib
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import generate_asset as ga  # api(), poll(), first_image(), download(), wait_for_server()


def upload_image(path):
    """POST an image to ComfyUI's input folder; returns the stored name."""
    name = f"bg_{random.randint(0, 1 << 30)}_{os.path.basename(path)}"
    boundary = "----assetforge"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
               f"filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n".encode())
    body.write(pathlib.Path(path).read_bytes())
    body.write(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
               f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{paths.SERVER}/upload/image", data=body.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["name"]


def build_workflow(image_name, model="birefnet.safetensors"):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": model}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        # RemoveBackground returns a foreground mask; JoinImageWithAlpha wants
        # it inverted, matching ComfyUI's own BiRefNet template.
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"filename_prefix": "bg_removed", "images": ["5", 0]}},
    }


def remove_background(input_path, output_path, crop=False, pad=8, model="birefnet.safetensors"):
    ga.wait_for_server()
    name = upload_image(input_path)
    entry = ga.poll(ga.submit(build_workflow(name, model)))
    ga.download(ga.first_image(entry), output_path)
    if crop:
        from PIL import Image
        im = Image.open(output_path).convert("RGBA")
        box = im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
        if box:
            l, t, r, b = box
            im = im.crop((max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad)))
            im.save(output_path)
        return im.size
    return None


def main():
    p = argparse.ArgumentParser(description="BiRefNet background removal through ComfyUI.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--crop", action="store_true", help="crop to the opaque content plus --pad")
    p.add_argument("--pad", type=int, default=8)
    p.add_argument("--model", default="birefnet.safetensors",
                   help="file in ComfyUI/models/background_removal")
    a = p.parse_args()
    t = time.time()
    size = remove_background(a.input, a.output, a.crop, a.pad, a.model)
    print(f"wrote {a.output}{' ' + 'x'.join(map(str, size)) if size else ''} in {time.time()-t:.1f}s")


if __name__ == "__main__":
    main()
