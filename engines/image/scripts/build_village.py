#!/usr/bin/env python3
"""Compose a village map out of the generated tileset.

Terrain is painted into a tile grid, boundaries are feathered with a fine ordered
dither that only bites into the edge the neighbour is on, then objects are
composited bottom-anchored and depth-sorted so southern sprites overlap northern
ones the way a top-down game expects.
"""
import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from autotile import canonical, tile_mask
from autotile import N as BIT_N, E as BIT_E, S as BIT_S, W as BIT_W
from autotile import NE as BIT_NE, SE as BIT_SE, SW as BIT_SW, NW as BIT_NW

ASSETS = str(paths.OUT)
TILES = str(paths.TILES)

TILE = 32
COLS, ROWS = 44, 32
SEED = 7

GRASS = ["grass", "grass2", "grass3"]

# Terrain regions are laid out on a tile grid, but a border that follows tile
# edges reads as graph paper. Instead each region is turned into a pixel-level
# coverage field, warped by one shared noise field and re-thresholded, which
# gives an irregular hand-drawn boundary that still lands on exact pixels.
#
# The noise field is shared across every terrain layer on purpose: the pond and
# the sand ring around it are derived from the same ellipse, so they have to be
# distorted identically or the shore detaches from the water.

# Order regions are painted in. Later entries win where regions overlap.
PAINT_ORDER = ["sand", "water", "soil", "dirt", "stone"]


def noise_field(shape, rng, cell=13, octaves=3):
    """Smooth wandering field in roughly -1..1, built from blurred white noise."""
    h, w = shape
    field = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    for i in range(octaves):
        step = max(2, cell // (2 ** i))
        coarse = rng.random((h // step + 2, w // step + 2)).astype(np.float32)
        img = Image.fromarray((coarse * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC)
        layer = np.asarray(img, dtype=np.float32) / 255.0
        field += (layer - 0.5) * 2.0 * amp
        amp *= 0.5
    return field / max(1e-6, np.abs(field).max())


def region_mask(region, noise, softness=9.0, wobble=0.42):
    """Pixel-resolution mask for one terrain region, with an organic edge.

    The tile-resolution boolean is blown up to pixels, blurred into a coverage
    ramp, pushed around by the shared noise, then re-thresholded. Blur width
    sets how far the border can wander; wobble sets how much it actually does.
    """
    big = np.kron(region.astype(np.float32), np.ones((TILE, TILE), dtype=np.float32))
    blurred = Image.fromarray((big * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(softness))
    coverage = np.asarray(blurred, dtype=np.float32) / 255.0
    return (coverage + noise * wobble) > 0.5


def load_tiles(tiles_dir=None):
    d = tiles_dir or TILES
    return {f[:-4]: Image.open(os.path.join(d, f)).convert("RGBA")
            for f in os.listdir(d) if f.endswith(".png")}


def ellipse(cx, cy, rx, ry):
    ys, xs = np.mgrid[0:ROWS, 0:COLS]
    return ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0


def grow(mask, n=1):
    out = mask.copy()
    for _ in range(n):
        p = np.pad(out, 1, constant_values=False)
        out = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    return out


def build_terrain():
    """Return a ROWS x COLS grid of terrain names."""
    terr = np.full((ROWS, COLS), "grass", dtype=object)

    pond = ellipse(36, 6, 6.5, 4.0)
    terr[grow(pond, 1) & ~pond] = "sand"
    terr[pond] = "water"

    terr[14:16, :] = "dirt"                 # main road, east-west
    terr[16:ROWS, 21:23] = "dirt"           # spur running south to the farm
    terr[10:14, 18:26] = "stone"            # village square
    terr[19:29, 27:41] = "soil"             # ploughed field
    return terr


NEIGHBOUR_BITS = [(-1, 0, BIT_N), (-1, 1, BIT_NE), (0, 1, BIT_E), (1, 1, BIT_SE),
                  (1, 0, BIT_S), (1, -1, BIT_SW), (0, -1, BIT_W), (-1, -1, BIT_NW)]


def neighbour_bits(region, y, x):
    """8-bit code for which neighbours share this cell's terrain.

    Off-map counts as matching, so regions run off the edge instead of being
    outlined against nothing.
    """
    bits = 0
    for dy, dx, bit in NEIGHBOUR_BITS:
        ny, nx = y + dy, x + dx
        if not (0 <= ny < ROWS and 0 <= nx < COLS) or region[ny, nx]:
            bits |= bit
    return bits


def paint_terrain(terr, tiles, rng, borders="noise"):
    canvas = Image.new("RGBA", (COLS * TILE, ROWS * TILE))

    def texture(name):
        """Fill the whole map with one terrain, varied so it does not repeat."""
        layer = Image.new("RGBA", canvas.size)
        for y in range(ROWS):
            for x in range(COLS):
                key = rng.choice(GRASS) if name == "grass" else name
                tile = tiles[key]
                if name in ("grass", "sand", "water"):
                    if rng.random() < 0.5:
                        tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
                    if rng.random() < 0.5:
                        tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
                layer.paste(tile, (x * TILE, y * TILE))
        return layer

    canvas.paste(texture("grass"), (0, 0))

    if borders == "autotile":
        # Engine-standard path: one lookup per cell into the 47-tile blob set.
        # Edges are crisp and, unlike the noise mode, the exact same tiles can
        # be exported as an atlas for Godot or Unity to draw at runtime.
        for name in PAINT_ORDER:
            region = (terr == name)
            if not region.any():
                continue
            layer = texture(name)
            for y in range(ROWS):
                for x in range(COLS):
                    if not region[y, x]:
                        continue
                    mask = tile_mask(canonical(neighbour_bits(region, y, x)))
                    patch = layer.crop((x * TILE, y * TILE, (x + 1) * TILE, (y + 1) * TILE))
                    canvas.paste(patch, (x * TILE, y * TILE),
                                 Image.fromarray((mask * 255).astype(np.uint8), "L"))
        return canvas

    noise = noise_field((ROWS * TILE, COLS * TILE), np.random.default_rng(SEED))
    for name in PAINT_ORDER:
        region = (terr == name)
        if not region.any():
            continue
        # The waterline is a real edge, so it wanders less than a path edge.
        wobble = 0.22 if name == "water" else 0.42
        mask = region_mask(region, noise, wobble=wobble)
        canvas.paste(texture(name), (0, 0),
                     Image.fromarray((mask * 255).astype(np.uint8), "L"))
    return canvas


def place(rng):
    """Build the object list as (name, tile_x, tile_y) bottom-centre anchors."""
    out = []

    def add(name, tx, ty):
        out.append((name, tx, ty))

    # Cottages lining the north side of the main road.
    for name, x in (("house_red", 5), ("house_blue", 11), ("house_red", 30), ("house_blue", 37)):
        add(name, x, 13)

    # Barn and its fenced paddock in the south-west.
    add("barn", 5, 22)
    for x in range(2, 11):
        add("fence", x, 26)
    for y in range(23, 26):
        add("fence", 2, y)
        add("fence", 10, y)
    add("haystack", 8, 24)
    add("haystack", 9, 25)

    # Village square: the well, ringed with lamps.
    add("well", 22, 13)
    for x, y in ((19, 11), (25, 11), (19, 13), (25, 13)):
        add("lamp", x, y)
    for x in range(3, 44, 7):
        add("lamp", x, 14)

    # Crop rows on the ploughed field, fenced on the south side.
    for y in range(20, 29, 2):
        for x in range(28, 41, 2):
            add("crop", x, y)
    for x in range(27, 41):
        add("fence", x, 29)

    # Forest edging the map, thinned so it frames rather than walls in the scene.
    for x in range(COLS):
        for y in (0, 1):
            if rng.random() < 0.55:
                add(rng.choice(["tree_oak", "tree_pine"]), x, y)
    for y in range(2, ROWS):
        for x in (0, 1, COLS - 2, COLS - 1):
            if rng.random() < 0.35:
                add(rng.choice(["tree_oak", "tree_pine"]), x, y)

    # Loose scatter on open grass.
    taken = {(x, y) for _, x, y in out}
    for _ in range(90):
        x, y = rng.randrange(2, COLS - 2), rng.randrange(2, ROWS - 1)
        if (x, y) in taken:
            continue
        taken.add((x, y))
        out.append((rng.choice(["bush", "rock", "tree_oak", "tree_pine"]), x, y))
    return out


def composite(canvas, placements, terr, tiles, ground_at=None):
    """Draw objects bottom-anchored, far to near, skipping unusable ground."""
    made_ground = {"water", "dirt", "stone", "soil"}
    drawn = 0
    for name, tx, ty in sorted(placements, key=lambda p: (p[2], p[1])):
        # Borders wander off the tile grid now, so ask what is actually painted
        # under the sprite's feet rather than what the layout grid intended.
        ground = ground_at(tx, ty) if ground_at else terr[ty, tx]
        if name in ("bush", "rock", "tree_oak", "tree_pine") and ground in made_ground:
            continue
        if name == "crop" and ground != "soil":
            continue
        sprite = tiles[name]
        canvas.alpha_composite(sprite, (tx * TILE + TILE // 2 - sprite.width // 2,
                                        (ty + 1) * TILE - sprite.height))
        drawn += 1
    return drawn


def main(tiles_dir=None, out_name="village_map.png", borders="noise"):
    rng = random.Random(SEED)
    tiles = load_tiles(tiles_dir)

    terr = build_terrain()
    canvas = paint_terrain(terr, tiles, rng, borders=borders)

    # Resolve what each cell ended up as after the borders were warped, by
    # re-deriving the same masks the painter used.
    painted = np.full((ROWS, COLS), "grass", dtype=object)
    if borders == "autotile":
        # Autotile edges stay inside their own cell, so the layout grid is
        # already the truth about what a sprite is standing on.
        painted = terr.copy()
    else:
        noise = noise_field((ROWS * TILE, COLS * TILE), np.random.default_rng(SEED))
        for name in PAINT_ORDER:
            region = (terr == name)
            if not region.any():
                continue
            wobble = 0.22 if name == "water" else 0.42
            mask = region_mask(region, noise, wobble=wobble)
            # A cell counts as this terrain when the footing is mostly on it.
            cell = mask.reshape(ROWS, TILE, COLS, TILE).mean(axis=(1, 3))
            painted[cell > 0.5] = name

    drawn = composite(canvas, place(rng), terr, tiles,
                      ground_at=lambda x, y: painted[y, x])

    out = os.path.join(ASSETS, out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.convert("RGB").save(out)
    print(f"map {canvas.width}x{canvas.height}px  ({COLS}x{ROWS} tiles)  {drawn} sprites -> {out}")

    crop = canvas.crop((15 * TILE, 8 * TILE, 31 * TILE, 20 * TILE))
    detail = os.path.join(ASSETS, f"{os.path.splitext(out_name)[0]}_detail.png")
    os.makedirs(os.path.dirname(detail), exist_ok=True)
    crop.resize((crop.width * 2, crop.height * 2), Image.NEAREST).convert("RGB").save(detail)
    print(f"detail 2x nearest -> {os.path.basename(detail)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compose the village map.")
    ap.add_argument("--tiles-dir", default=None)
    ap.add_argument("--out", default="village_map.png")
    ap.add_argument("--borders", choices=("noise", "autotile"), default="noise",
                    help="noise-warped organic edges, or the 47-tile blob set")
    a = ap.parse_args()
    sys.exit(main(a.tiles_dir, a.out, a.borders))
