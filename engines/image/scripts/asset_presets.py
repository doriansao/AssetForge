#!/usr/bin/env python3
"""Recipes for 2D game assets across genres, and a CLI to render them.

The tileset scripts in this folder are one workflow. Most of what a 2D game
needs is not a top-down terrain tile: a platformer wants side-on ground blocks
and props, a beat 'em up wants full-body character sprites, every genre wants
inventory icons and UI. Those differ in three ways that matter, and a preset
pins all three so the caller only supplies a subject:

  framing      a top-down tile is seen from above and fills the frame; a
               platformer prop is seen side-on and must be a cutout; a
               character needs its whole body in shot with room around it
  seamlessness terrain repeats and must wrap; a prop must not
  resolution   an inventory icon lives at 32px, a boss sprite at 200px, and
               the right post-process differs with it

The post-processing choice per preset is the part worth reading. "snap" runs
the grid snapper, which recovers the block grid Flux drew and is right when a
sprite may keep its own natural resolution. "box" is a plain area downscale,
which is right when the sprite must land on an exact size, because snapping and
then resizing destroys the grid the snapper just recovered. See PIPELINE.md.

Usage:
  python asset_presets.py --list
  python asset_presets.py --preset platformer_tile --subject "mossy stone block" \
      --output assets/platformer/stone.png
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import pixel_art_processor as pap
from generate_asset import (build_workflow, download, first_image, poll, submit,
                            wait_for_server)

# Era vocabulary. Kept separate from the presets so any subject can be rendered
# in any era without duplicating every recipe.
ERAS = {
    "8bit": "8-bit NES era sprite, 3 to 4 colours per sprite, bold 1 pixel dark outlines, chunky readable shapes",
    "16bit": "16-bit SNES era sprite, around 16 to 24 colours, 2 to 3 tones per region, 1 pixel dark outlines",
    "gba": "GBA era handheld sprite, around 24 colours, soft shading with selective dithering",
    "modern": "modern indie pixel art, around 32 colours, painterly shading within a strict pixel grid",
}

# White, and keyed with a loose flood fill. Models draw a contact shadow
# whatever the prompt says, and Flux Schnell ignores negative prompts entirely,
# so it cannot be prevented at generation time. It is removed instead by keying
# wide enough to walk from the white background into the soft grey of the
# shadow; see key_background_to_alpha.
#
# A magenta chroma key was tried and reverted. It removes shadows perfectly,
# because a shadow on the key is a darker shade of the key. But magenta's strong
# channels are red and blue, and this kind of palette has both -- warm wood and
# blue-grey stone -- so the keyer punched holes through crates and left magenta
# fringing on stone. pixel_art_processor.key_chroma_to_alpha is still there for
# palettes that genuinely suit one.
CUTOUT = ("isolated as a cutout on a plain solid pure white background, "
          "no shadow, no ground patch, no platform, no text")

# framing:   how the subject is described and posed
# size:      render resolution handed to Flux
# post:      "snap" (grid snapper), "box" (area downscale), or "tile" (seamless)
# target:    longest side of the finished sprite, or None to keep native
# palette:   k-means colour count
PRESETS = {
    "topdown_tile": dict(
        framing="top-down seamless ground texture of {s}, fills the entire frame, no objects, no shadows",
        size=512, post="tile", target=32, palette=24, seamless=True,
        note="Terrain for a top-down map. Wrap-blended so it repeats."),
    "topdown_object": dict(
        framing="{s}, seen from above at a slight angle, single game sprite centred in frame, " + CUTOUT,
        size=768, post="box", target=64, palette=24,
        note="Trees, buildings, props for a top-down map."),
    "platformer_tile": dict(
        framing="side-on seamless platform ground texture of {s}, straight-on flat view, "
                "fills the entire frame, no objects, no perspective",
        size=512, post="tile", target=32, palette=24, seamless=True,
        note="Side-on ground/wall block for a platformer. Wraps on both axes."),
    "platformer_prop": dict(
        framing="{s}, straight-on side view at eye level, single game sprite centred in frame, " + CUTOUT,
        size=768, post="box", target=48, palette=24,
        note="Crates, barrels, spikes, signs seen side-on."),
    "character": dict(
        framing="a full body {s}, straight-on side view facing right, standing idle pose, "
                "entire body visible from head to feet with clear space around it, " + CUTOUT,
        size=768, post="snap", target=None, palette=32,
        note="Beat 'em up / platformer character. Keeps its native resolution; "
             "let the snapper decide the size and scale in the engine."),
    "character_portrait": dict(
        framing="a head and shoulders portrait of {s}, facing the viewer, " + CUTOUT,
        size=768, post="snap", target=None, palette=32,
        note="Dialogue portrait. Higher palette because skin needs several tones."),
    "item_icon": dict(
        framing="{s}, single inventory item icon, straight-on view, centred, " + CUTOUT,
        size=1024, post="snap", target=32, palette=16,
        note="Inventory / UI icon. Small and flat by design."),
    "enemy": dict(
        framing="a full body {s} enemy creature, straight-on side view facing right, "
                "menacing idle pose, entire body visible, " + CUTOUT,
        size=768, post="snap", target=None, palette=24,
        note="Side-on enemy sprite for a platformer or beat 'em up."),
    "parallax_layer": dict(
        framing="a wide side-on parallax background layer of {s}, flat silhouetted depth layer, "
                "horizontally seamless, no foreground objects",
        size=1024, post="tile", target=None, palette=24, seamless="horizontal",
        note="Scrolling background band. Wraps horizontally only."),
}


def render(prompt, size, seed, lora=None, lora_strength=1.0, model=None):
    """Run one generation and return the full-resolution frame."""
    wf = build_workflow(prompt, size, size, seed, lora=lora,
                        lora_strength=lora_strength, model=model)
    entry = poll(submit(wf))
    tmp = os.path.join(os.environ.get("TEMP", "."), f"_preset_{seed}.png")
    download(first_image(entry), tmp)
    img = Image.open(tmp).copy()
    os.remove(tmp)
    return img


def wrap_blend(img, frac=0.25, axes=(0, 1)):
    """Cross-fade edges into their opposites so the texture tiles."""
    import numpy as np
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    for axis in axes:
        a = np.moveaxis(a, axis, 0)
        n = a.shape[0]
        m = int(n * frac)
        ramp = np.linspace(0.0, 1.0, m).reshape(m, *([1] * (a.ndim - 1)))
        out = a[: n - m].copy()
        out[:m] = a[:m] * ramp + a[n - m:] * (1.0 - ramp)
        a = np.moveaxis(out, 0, axis)
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def build(preset_name, subject, era="16bit", seed=None, lora=None,
          lora_strength=1.0, model=None):
    if preset_name not in PRESETS:
        raise SystemExit(f"unknown preset {preset_name!r}; try --list")
    p = PRESETS[preset_name]
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    prompt = f"{p['framing'].format(s=subject)}, {ERAS[era]}, pixel art"
    raw = render(prompt, p["size"], seed, lora, lora_strength, model)

    if p["post"] == "tile":
        axes = (1,) if p.get("seamless") == "horizontal" else (0, 1)
        blended = wrap_blend(raw, axes=axes)
        if p["target"]:
            side = p["target"]
            sprite = blended.resize((side, side), Image.BOX).convert("RGBA")
        else:
            sprite = blended.convert("RGBA")
        info = {"final_size": sprite.size, "native_size": blended.size,
                "detected_step": 0.0, "colors": p["palette"], "sprite": sprite}
    elif p["post"] == "snap":
        info = pap.process(raw, target_size=p["target"], palette_size=p["palette"])
    else:  # box
        keyed = pap.key_background_to_alpha(raw)
        keyed = pap.crop_to_content(pap.trim_ground_plinth(keyed))
        side = p["target"]
        scale = side / max(keyed.size)
        info = {"sprite": keyed.resize((max(1, round(keyed.width * scale)),
                                        max(1, round(keyed.height * scale))), Image.BOX),
                "native_size": keyed.size, "detected_step": 0.0, "colors": p["palette"]}
        info["final_size"] = info["sprite"].size
    info["raw"] = raw
    info["seed"] = seed
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the presets and exit")
    ap.add_argument("--preset")
    ap.add_argument("--subject", help="what to draw, e.g. 'a mossy stone block'")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--era", default="16bit", choices=sorted(ERAS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-strength", type=float, default=0.9)
    ap.add_argument("--model", default=None, help="image model; default flux-schnell")
    ap.add_argument("--keep-raw", action="store_true", help="also save the full-resolution frame")
    ap.add_argument("--normal-map", action="store_true", help="also write a normal map")
    args = ap.parse_args()

    if args.list:
        width = max(len(k) for k in PRESETS)
        for name, p in PRESETS.items():
            print(f"{name:{width}}  {p['size']}px -> "
                  f"{str(p['target']) + 'px' if p['target'] else 'native'}, "
                  f"{p['palette']} colours, post={p['post']}")
            print(f"{'':{width}}  {p['note']}")
        return 0

    if not (args.preset and args.subject and args.output):
        raise SystemExit("--preset, --subject and --output are all required")

    wait_for_server()
    info = build(args.preset, args.subject, args.era, args.seed, args.lora,
                 args.lora_strength, args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    info["sprite"].save(args.output)
    print(f"{args.preset}: {info['final_size'][0]}x{info['final_size'][1]}"
          f"  seed {info['seed']}  -> {args.output}")

    if args.keep_raw:
        raw_path = args.output.with_name(f"{args.output.stem}_raw.png")
        info["raw"].save(raw_path)
        print(f"  raw -> {raw_path.name}")
    if args.normal_map:
        from make_normalmap import make_normal
        n_path = args.output.with_name(f"{args.output.stem}_n.png")
        make_normal(info["sprite"]).save(n_path)
        print(f"  normal -> {n_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
