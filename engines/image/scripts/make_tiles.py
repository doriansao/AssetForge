#!/usr/bin/env python3
"""Turn the raw Flux renders into a coherent pixel-art tileset.

Flux paints at 512-768px with soft edges and thousands of colours, which is not
pixel art. This stage does the conversion:

  terrain  -> wrap-blended so it repeats without a seam, box-downsampled to 32px
  objects  -> white background keyed out, cropped to content, scaled to a sprite box
  both     -> snapped to one shared palette so every tile looks like one artist

The shared palette is the important part: quantising each tile on its own gives
18 unrelated colour sets, and the assembled map looks like a collage.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import pixel_art_processor as pap

ASSETS = str(paths.OUT)
RAW = str(paths.RAW)
TILES = str(paths.TILES)

TILE = 32
# 32 beat both 128 and a fixed DawnBringer-32 in a three-way comparison. 128
# left AI-ish smooth gradients in the grass; DB32 has too few greens for this
# art and speckled the fields. 32 adaptive colours flatten the texture into
# readable blocks while keeping the palette derived from the render itself.
PALETTE_SIZE = 32

# DawnBringer 32, the most widely used hand-designed pixel-art palette
# (published on Lospec). Hand-designed palettes carry deliberate hue shifts --
# shadows lean blue/purple, highlights lean yellow -- which is the thing an
# adaptive palette derived from the art itself can never invent, because it
# only ever redistributes colours the render already had.
DB32 = [
    "000000", "222034", "45283c", "663931", "8f563b", "df7126", "d9a066", "eec39a",
    "fbf236", "99e550", "6abe30", "37946e", "4b692f", "524b24", "323c39", "3f3f74",
    "306082", "5b6ee1", "639bff", "5fcde4", "cbdbfc", "ffffff", "9badb7", "847e87",
    "696a6a", "595652", "76428a", "ac3232", "d95763", "d77bba", "8f974a", "8a6f30",
]

# Measured worse than a plain box downsample for THIS pipeline, so it is off.
# The snapper recovers the block grid Flux drew, but tileset sprites then have
# to be resized onto fixed box sizes, and that second resample destroys the
# grid again -- the barn lost its roof, the lamp broke apart. It wins only when
# a sprite is allowed to keep its own detected resolution, as standalone icons
# from generate_asset.py --pixel-art do.
SNAP_SPRITES = False
SNAP_COLORS = 24

TERRAIN = ["grass", "grass2", "grass3", "dirt", "water", "stone", "sand", "soil"]

# name -> (max width, max height) in pixels, aspect preserved inside the box
OBJECTS = {
    "tree_oak":   (64, 72, False),
    "tree_pine":  (56, 80, False),
    "house_red":  (128, 120, False),
    "house_blue": (128, 120, False),
    "barn":       (150, 130, False),
    "well":       (56, 56, False),
    "bush":       (32, 32, False),
    "rock":       (32, 28, False),
    "fence":      (32, 24, True),
    "crop":       (28, 28, False),
    "lamp":       (26, 62, False),
    "haystack":   (44, 40, False),
}


def wrap_blend(img, frac=0.25):
    """Cross-fade each edge into the opposite one so the texture tiles seamlessly.

    Result column 0 continues from what was column W-m, so the wrap is smooth.
    """
    a = np.asarray(img, dtype=np.float32)
    for axis in (0, 1):
        a = np.moveaxis(a, axis, 0)
        n = a.shape[0]
        m = int(n * frac)
        ramp = np.linspace(0.0, 1.0, m).reshape(m, *([1] * (a.ndim - 1)))
        out = a[: n - m].copy()
        out[:m] = a[:m] * ramp + a[n - m:] * (1.0 - ramp)
        a = np.moveaxis(out, 0, axis)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def key_out_background(img, thresh=72, drop_enclosed=False):
    """Flood the plain white field in from the border and turn it into alpha.

    Flood fill rather than a plain white threshold, so white pixels *inside* the
    sprite (window frames, highlights) survive.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    sentinel = (255, 0, 255)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
             (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    for seed in seeds:
        if sum(rgb.getpixel(seed)) > 700 - thresh * 3:
            ImageDraw.floodfill(rgb, seed, sentinel, thresh=thresh)

    arr = np.asarray(rgb)
    bg = np.all(arr == np.array(sentinel), axis=-1)
    if drop_enclosed:
        bg |= np.all(arr > 255 - thresh, axis=-1)

    out = np.dstack([arr, np.where(bg, 0, 255).astype(np.uint8)])
    out[bg] = (0, 0, 0, 0)
    return Image.fromarray(out, "RGBA")


def content_box(rgba):
    alpha = np.asarray(rgba)[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def punch(img, sat=1.35, con=1.12):
    """Pixel art reads better with more saturation and contrast than Flux gives."""
    img = ImageEnhance.Color(img).enhance(sat)
    return ImageEnhance.Contrast(img).enhance(con)


def match_tone(img, ref, strength=0.85):
    """Pull an image's per-channel mean and spread towards a reference tile.

    The three grass variants came out of Flux at noticeably different brightness,
    which tiled into a visible patchwork. Matching tone keeps the texture variety
    while removing the patchiness.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    r = np.asarray(ref.convert("RGB"), dtype=np.float32)
    for c in range(3):
        am, asd = a[..., c].mean(), a[..., c].std() + 1e-6
        rm, rsd = r[..., c].mean(), r[..., c].std() + 1e-6
        matched = (a[..., c] - am) * (rsd / asd) + rm
        a[..., c] = a[..., c] * (1 - strength) + matched * strength
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def build_terrain():
    out = {}
    for name in TERRAIN:
        src = Image.open(os.path.join(RAW, f"{name}.png")).convert("RGB")
        tile = wrap_blend(src).resize((TILE, TILE), Image.BOX)
        tile = punch(tile, sat=1.12, con=1.05)
        if name in ("grass2", "grass3"):
            tile = match_tone(tile, out["grass"])
        out[name] = tile.convert("RGBA")
        print(f"[terrain] {name:6s} -> {TILE}x{TILE} seamless", flush=True)
    return out


def build_objects():
    out = {}
    for name, (bw, bh, drop) in OBJECTS.items():
        src = Image.open(os.path.join(RAW, f"{name}.png")).convert("RGB")
        # Key out at a quarter size: the flood fill is far cheaper and the sprite
        # is being shrunk to well under this anyway.
        small = src.resize((256, 256), Image.BOX)
        rgba = key_out_background(small, drop_enclosed=drop)

        box = content_box(rgba)
        if box is None:
            print(f"[WARN] {name}: background fill removed everything, using full frame")
            rgba = small.convert("RGBA")
            box = (0, 0, 256, 256)
        rgba = rgba.crop(box)

        scale = min(bw / rgba.width, bh / rgba.height)
        size = (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale)))

        if SNAP_SPRITES:
            # Let the grid snapper find the block size Flux actually drew and
            # take one colour per block, instead of averaging across blocks the
            # way a plain downscale does. Averaging is what turns a one-pixel
            # outline into a three-pixel gradient.
            snapped = pap.snap_to_grid(rgba, colors=SNAP_COLORS)[0]
            sprite = snapped.resize(size, Image.NEAREST)
        else:
            sprite = rgba.resize(size, Image.BOX)

        arr = np.asarray(sprite).copy()
        arr[..., 3] = np.where(arr[..., 3] > 128, 255, 0)  # binary alpha, no soft halo
        sprite = Image.fromarray(arr, "RGBA")

        rgb = punch(sprite.convert("RGB"), sat=1.22, con=1.08)
        sprite = Image.merge("RGBA", (*rgb.split(), sprite.split()[3]))
        out[name] = sprite
        print(f"[object ] {name:11s} -> {size[0]}x{size[1]}", flush=True)
    return out


def fixed_palette(hexes):
    """Build a PIL palette image from a list of hex colours."""
    rgb = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in hexes]
    pal = Image.new("P", (1, 1))
    flat = [c for colour in rgb for c in colour]
    pal.putpalette(flat + [0, 0, 0] * (256 - len(rgb)))
    return pal


def shared_palette(images, size=PALETTE_SIZE):
    """One adaptive palette derived from every tile, so the set stays cohesive."""
    swatches = []
    for img in images:
        rgb = img.convert("RGBA")
        arr = np.asarray(rgb)
        opaque = arr[arr[..., 3] > 0][:, :3]
        if len(opaque) == 0:
            continue
        # Repeat each tile's pixels equally so a big sprite cannot dominate.
        idx = np.random.default_rng(0).choice(len(opaque), size=min(4096, len(opaque)), replace=False)
        swatches.append(opaque[idx])
    pool = np.concatenate(swatches)
    side = int(np.ceil(np.sqrt(len(pool))))
    padded = np.zeros((side * side, 3), dtype=np.uint8)
    padded[: len(pool)] = pool
    return Image.fromarray(padded.reshape(side, side, 3), "RGB").quantize(
        colors=size, method=Image.MEDIANCUT)


def snap(img, pal):
    """Map an RGBA sprite onto the shared palette, preserving its alpha."""
    alpha = img.convert("RGBA").split()[3]
    flat = Image.new("RGB", img.size, (255, 255, 255))
    flat.paste(img.convert("RGB"), (0, 0), alpha)
    quant = flat.quantize(palette=pal, dither=Image.NONE).convert("RGB")
    return Image.merge("RGBA", (*quant.split(), alpha))


def main(palette_mode="adaptive", out_dir=None):
    out_dir = out_dir or TILES
    os.makedirs(out_dir, exist_ok=True)
    missing = [n for n in TERRAIN + list(OBJECTS) if not os.path.exists(os.path.join(RAW, f"{n}.png"))]
    if missing:
        raise SystemExit(f"missing raw renders: {missing}\nrun gen_tileset.py first")

    terrain = build_terrain()
    objects = build_objects()

    if palette_mode == "db32":
        pal = fixed_palette(DB32)
        label = f"DawnBringer 32 (fixed)"
    else:
        size = PALETTE_SIZE if palette_mode == "adaptive" else int(palette_mode.replace("adaptive", ""))
        pal = shared_palette(list(terrain.values()) + list(objects.values()), size)
        label = f"{size} adaptive"
    print(f"\n[palette] {label}")

    for name, img in {**terrain, **objects}.items():
        snap(img, pal).save(os.path.join(out_dir, f"{name}.png"))
    print(f"[write  ] {len(terrain)+len(objects)} tiles -> {out_dir}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the pixel-art tileset.")
    ap.add_argument("--palette", default="adaptive",
                    help="adaptive (128), adaptive32, or db32")
    ap.add_argument("--tiles-dir", default=None)
    a = ap.parse_args()
    sys.exit(main(a.palette, a.tiles_dir))
