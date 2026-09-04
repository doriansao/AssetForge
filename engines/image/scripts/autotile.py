#!/usr/bin/env python3
"""Build a 47-tile blob autotile set from two terrain tiles.

This is what shipping 2D engines use for terrain transitions: instead of one
tile per terrain plus a blended edge, you author every combination of "which of
my 8 neighbours are the same terrain as me" and look the right one up per cell.
Godot 4 calls it a terrain set, Unity calls it a rule tile.

The 256 possible 8-bit neighbourhoods collapse to 47 distinct tiles, because a
diagonal neighbour only matters when both of its adjacent sides are also the
same terrain. Rather than author 47 tiles by hand, each is assembled from four
16x16 quadrants, and a quadrant has only five possible shapes:

    fill          both sides and the diagonal match, so the terrain is solid
    outer corner  neither side matches, terrain rounds away from the corner
    side edge     one side matches, terrain is cut back along a straight line
    inner corner  both sides match but the diagonal does not, small notch

Five shapes times four quadrants reproduces the full 47-tile set exactly, and
every tile stays consistent with its neighbours by construction.

Compared with the noise-warped borders in build_village, autotiling gives
crisp deliberate edges and, more importantly, exports as a real atlas plus a
lookup table that an engine can consume directly.

Usage:
  python autotile.py --base grass --over dirt --out assets/autotiles
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

TILE = 32
HALF = TILE // 2

# Bit positions of the 8 neighbours, clockwise from north.
N, NE, E, SE, S, SW, W, NW = (1 << i for i in range(8))

# Each quadrant is described by its two adjacent sides and the diagonal between
# them, given here as (side_a_bit, diagonal_bit, side_b_bit) with side_a running
# vertically and side_b horizontally.
QUADRANTS = {
    "tl": (N, NW, W),
    "tr": (N, NE, E),
    "bl": (S, SW, W),
    "br": (S, SE, E),
}


def quadrant_shape(vertical: bool, diagonal: bool, horizontal: bool,
                   radius: float = 6.0, notch: float = 4.0) -> np.ndarray:
    """Coverage mask for one 16x16 quadrant, corner of the tile at (0, 0)."""
    u, v = np.meshgrid(np.arange(HALF), np.arange(HALF), indexing="xy")
    dist = np.hypot(u, v)

    if vertical and horizontal:
        # Both sides continue. Only a missing diagonal cuts anything, and then
        # only a small notch right at the corner.
        return dist >= notch if not diagonal else np.ones_like(dist, dtype=bool)
    if vertical and not horizontal:
        return u >= radius          # cut back along the horizontal neighbour
    if horizontal and not vertical:
        return v >= radius          # cut back along the vertical neighbour
    return dist >= radius           # outer corner, rounds away


def tile_mask(bits: int) -> np.ndarray:
    """Full 32x32 coverage mask for one neighbourhood bitmask."""
    mask = np.zeros((TILE, TILE), dtype=bool)
    placement = {"tl": (0, 0), "tr": (0, HALF), "bl": (HALF, 0), "br": (HALF, HALF)}
    for key, (side_v, diag, side_h) in QUADRANTS.items():
        quad = quadrant_shape(bool(bits & side_v), bool(bits & diag), bool(bits & side_h))
        # Each quadrant is built with its outer corner at the origin, so mirror
        # it into place rather than recomputing four near-identical shapes.
        if key in ("tr", "br"):
            quad = quad[:, ::-1]
        if key in ("bl", "br"):
            quad = quad[::-1, :]
        y, x = placement[key]
        mask[y:y + HALF, x:x + HALF] = quad
    return mask


def canonical(bits: int) -> int:
    """Drop diagonal bits that cannot matter, collapsing 256 cases to 47.

    A diagonal only affects the picture when both of its adjacent sides are also
    the same terrain; otherwise that corner is already cut back by a side.
    """
    out = bits & (N | E | S | W)
    for diag, (a, b) in ((NE, (N, E)), (SE, (S, E)), (SW, (S, W)), (NW, (N, W))):
        if bits & diag and bits & a and bits & b:
            out |= diag
    return out


def build_set(base: Image.Image, over: Image.Image) -> dict[int, Image.Image]:
    """One composited tile per distinct neighbourhood."""
    tiles: dict[int, Image.Image] = {}
    for bits in range(256):
        key = canonical(bits)
        if key in tiles:
            continue
        mask = Image.fromarray((tile_mask(key) * 255).astype(np.uint8), "L")
        tile = base.copy()
        tile.paste(over, (0, 0), mask)
        tiles[key] = tile
    return tiles


def export_atlas(tiles: dict[int, Image.Image], out_dir: str, name: str) -> str:
    """Write an atlas PNG plus the bitmask lookup an engine needs."""
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted(tiles)
    cols = 8
    rows = (len(keys) + cols - 1) // cols
    atlas = Image.new("RGBA", (cols * TILE, rows * TILE))
    lookup = {}
    for i, key in enumerate(keys):
        cx, cy = i % cols, i // cols
        atlas.paste(tiles[key], (cx * TILE, cy * TILE))
        lookup[key] = {"index": i, "x": cx, "y": cy}

    atlas_path = os.path.join(out_dir, f"{name}_atlas.png")
    atlas.save(atlas_path)
    with open(os.path.join(out_dir, f"{name}_atlas.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "tile_size": TILE,
            "columns": cols,
            "rows": rows,
            "tile_count": len(keys),
            "bit_order": "N,NE,E,SE,S,SW,W,NW as bits 0..7",
            "note": "look up canonical(neighbour_bits); diagonals only count "
                    "when both adjacent sides match",
            "tiles": {str(k): v for k, v in lookup.items()},
        }, fh, indent=2)
    return atlas_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles-dir", default=str(paths.TILES))
    ap.add_argument("--base", required=True, help="terrain underneath, e.g. grass")
    ap.add_argument("--over", required=True, help="terrain painted on top, e.g. dirt")
    ap.add_argument("--out", default=str(paths.OUT / "autotiles"))
    args = ap.parse_args()

    base = Image.open(os.path.join(args.tiles_dir, f"{args.base}.png")).convert("RGBA")
    over = Image.open(os.path.join(args.tiles_dir, f"{args.over}.png")).convert("RGBA")

    tiles = build_set(base, over)
    name = f"{args.over}_on_{args.base}"
    path = export_atlas(tiles, args.out, name)
    print(f"{len(tiles)} distinct tiles -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
