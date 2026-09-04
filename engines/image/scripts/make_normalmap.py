#!/usr/bin/env python3
"""Derive a tangent-space normal map from a finished sprite.

Modern 2D games light flat sprites with a normal map: the engine samples it per
pixel to work out which way the surface faces, so a torch moving past a wall
actually rakes across its bricks. Godot takes one via CanvasTexture's normal
map slot, Unity via the secondary texture on a Sprite Renderer with a 2D light.

There is no real depth in a painted sprite, so height is inferred from two
sources and combined:

  shading   luminance stands in for height, which recovers surface detail like
            plank grooves, roof tiles and brickwork
  silhouette a distance falloff inside the alpha edge, which rounds the sprite
            off at its outline instead of leaving it a flat card

Luminance alone gives a normal map that reads dark paint as a dent, so a black
outline becomes a trench. The silhouette term is what keeps the sprite reading
as a solid object.

Usage:
  python make_normalmap.py -i sprite.png -o sprite_n.png
  python make_normalmap.py -i tiles/ -o normals/ --strength 2.5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def height_field(img: Image.Image, bevel: float = 3.0, detail: float = 0.65) -> np.ndarray:
    """Blend a luminance height with a rounded-off silhouette bevel."""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3]

    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    luma = np.where(alpha > 0, luma, 0.0)

    # Blurring the alpha gives a cheap inward falloff: full inside the shape,
    # tapering to zero at the outline. That is the bevel.
    blur = Image.fromarray((alpha * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(bevel))
    silhouette = np.asarray(blur, dtype=np.float32) / 255.0
    silhouette = np.minimum(silhouette, alpha)  # never bulge outside the sprite

    return detail * luma + (1.0 - detail) * silhouette


def to_normal(height: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """Sobel the height field and encode the surface normal into RGB."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T

    pad = np.pad(height, 1, mode="edge")
    gx = sum(kx[j, i] * pad[j:j + height.shape[0], i:i + height.shape[1]]
             for j in range(3) for i in range(3))
    gy = sum(ky[j, i] * pad[j:j + height.shape[0], i:i + height.shape[1]]
             for j in range(3) for i in range(3))

    # X points right, Y points up in the OpenGL convention both Godot and Unity
    # expect, so the vertical gradient is negated.
    nx, ny, nz = -gx * strength, gy * strength, np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / length, ny / length, nz / length

    return np.dstack([(nx + 1) * 0.5, (ny + 1) * 0.5, (nz + 1) * 0.5])


def make_normal(img: Image.Image, strength: float = 2.0, bevel: float = 3.0,
                detail: float = 0.65) -> Image.Image:
    rgba = img.convert("RGBA")
    normal = to_normal(height_field(rgba, bevel, detail), strength)
    out = (np.clip(normal, 0, 1) * 255).astype(np.uint8)

    alpha = np.asarray(rgba)[..., 3]
    # Flat-facing normal outside the sprite, so bleeding never tilts anything.
    out[alpha == 0] = (128, 128, 255)
    return Image.fromarray(np.dstack([out, alpha]), "RGBA")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a normal map from a sprite.")
    ap.add_argument("--input", "-i", type=Path, required=True, help="PNG file or directory")
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--strength", type=float, default=2.0, help="surface relief (default 2.0)")
    ap.add_argument("--bevel", type=float, default=3.0, help="silhouette rounding in px")
    ap.add_argument("--detail", type=float, default=0.65,
                    help="0 = pure silhouette bevel, 1 = pure luminance detail")
    args = ap.parse_args()

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        made = 0
        for src in sorted(args.input.glob("*.png")):
            normal = make_normal(Image.open(src), args.strength, args.bevel, args.detail)
            normal.save(args.output / f"{src.stem}_n.png")
            made += 1
        print(f"{made} normal maps -> {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        make_normal(Image.open(args.input), args.strength, args.bevel, args.detail).save(args.output)
        print(f"normal map -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
