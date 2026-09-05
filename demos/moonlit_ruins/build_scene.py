#!/usr/bin/env python3
"""Compose a platformer scene from the Moonlit Ruins pack.

A demonstration that the pieces fit together: parallax background, tiled ground,
floating platforms, props, characters and a HUD, all from assets the engine
generated independently. Nothing here is drawn by hand.

The scene is laid out at native resolution on a 32px tile grid and upscaled with
NEAREST at the end, which is how a real 2D game presents pixel art. Composing at
the final size instead would put sprites on half-pixels and the whole thing
would read as blurry.

  python demos/moonlit_ruins/build_scene.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
PACK = ROOT / "out" / "image" / "moonlit_ruins"
OUT = ROOT / "out" / "image" / "moonlit_ruins"

TILE = 32
COLS, ROWS = 20, 12          # 640x384 native
SCALE = 2

GROUND_ROW = 8               # first solid row
SKY_TOP = (18, 22, 46)
SKY_BOTTOM = (58, 74, 112)
VOID = (10, 12, 24)          # what you see down a pit: not sky


def load(name: str) -> Image.Image:
    return Image.open(PACK / f"{name}.png").convert("RGBA")


def sky(size: tuple[int, int]) -> Image.Image:
    """Vertical gradient, so the background is not a flat field behind the art."""
    w, h = size
    top, bottom = np.array(SKY_TOP, np.float32), np.array(SKY_BOTTOM, np.float32)
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    band = top[None, :] * (1 - ramp) + bottom[None, :] * ramp
    return Image.fromarray(np.repeat(band[:, None, :], w, axis=1).astype(np.uint8), "RGB").convert("RGBA")


def parallax(canvas: Image.Image, layer: Image.Image, y: int, opacity: float) -> None:
    """Tile a background band horizontally and blend it back into the sky.

    Real parallax layers are drawn at reduced contrast so foreground art stays
    readable; opacity here stands in for that.
    """
    scale = (ROWS * TILE * 0.55) / layer.height
    band = layer.resize((max(1, int(layer.width * scale)), int(layer.height * scale)), Image.NEAREST)
    strip = Image.new("RGBA", (canvas.width + band.width, band.height))
    for x in range(0, strip.width, band.width):
        strip.paste(band, (x, 0))
    faded = Image.new("RGBA", strip.size)
    faded = Image.blend(faded, strip, opacity)
    canvas.alpha_composite(faded.crop((0, 0, canvas.width, band.height)), (0, y))


# Ledges as (x0, x1, row). Kept clear of where the actors stand: a two-tile-tall
# character bottom-anchored on the floor occupies the row above it too, so a
# ledge sharing that row cuts through the sprite.
LEDGES = ((3, 7, 5), (12, 16, 4))
PIT = (9, 12)


def solid(grid: np.ndarray) -> None:
    """Mark the level geometry: floor, a pit, and two floating ledges."""
    grid[GROUND_ROW:, :] = 1
    grid[GROUND_ROW:, PIT[0]:PIT[1]] = 0
    for x0, x1, y in LEDGES:
        grid[y, x0:x1] = 1


def draw_terrain(canvas: Image.Image, grid: np.ndarray,
                 ground: Image.Image, under: Image.Image) -> None:
    """Top of a solid run gets the surface tile; anything buried gets fill.

    Empty cells below the ground line are painted as void rather than left as
    sky, so a pit reads as depth instead of as a hole punched through to the
    background.
    """
    void = Image.new("RGBA", (TILE, TILE), VOID)
    for y in range(GROUND_ROW, ROWS):
        for x in range(COLS):
            if not grid[y, x]:
                canvas.alpha_composite(void, (x * TILE, y * TILE))

    for y in range(ROWS):
        for x in range(COLS):
            if not grid[y, x]:
                continue
            exposed = y == 0 or not grid[y - 1, x]
            canvas.alpha_composite(ground if exposed else under, (x * TILE, y * TILE))


def place(canvas: Image.Image, sprite: Image.Image, tx: float, ty: int,
          flip: bool = False) -> None:
    """Bottom-centre anchor on a tile, which is how a 2D engine positions actors."""
    if flip:
        sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)
    x = int(tx * TILE + TILE // 2 - sprite.width // 2)
    y = (ty + 1) * TILE - sprite.height
    canvas.alpha_composite(sprite, (x, y))


def hud(canvas: Image.Image, icons: list[Image.Image]) -> None:
    """Health pips and a small item tray, drawn from the generated icons."""
    for i in range(5):
        pip = Image.new("RGBA", (10, 10), (214, 74, 86, 255) if i < 4 else (60, 40, 52, 255))
        canvas.alpha_composite(pip, (10 + i * 13, 10))

    x = 10
    for icon in icons:
        fitted = icon.resize((20, max(1, round(20 * icon.height / icon.width))), Image.NEAREST)
        canvas.alpha_composite(fitted, (x, 26))
        x += 26


def main() -> int:
    missing = [n for n in ("ground", "underground", "crate", "brazier", "pillar",
                           "hero", "enemy", "icon_potion", "icon_key", "icon_gem", "bg_far")
               if not (PACK / f"{n}.png").exists()]
    if missing:
        raise SystemExit(f"missing pack assets: {missing}\nrun the pack generation first")

    canvas = sky((COLS * TILE, ROWS * TILE))
    parallax(canvas, load("bg_far"), y=TILE, opacity=0.55)

    grid = np.zeros((ROWS, COLS), dtype=np.uint8)
    solid(grid)
    draw_terrain(canvas, grid, load("ground"), load("underground"))

    # Props stand on whatever surface is beneath them: the brazier on the left
    # ledge, everything else on the floor.
    place(canvas, load("brazier"), 4.5, LEDGES[0][2] - 1)
    place(canvas, load("crate"), 1.0, GROUND_ROW - 1)
    place(canvas, load("crate"), 2.0, GROUND_ROW - 1)
    place(canvas, load("pillar"), 18.4, GROUND_ROW - 1)

    place(canvas, load("hero"), 6.5, GROUND_ROW - 1)
    place(canvas, load("enemy"), 14.6, GROUND_ROW - 1, flip=True)

    hud(canvas, [load("icon_potion"), load("icon_key"), load("icon_gem")])

    out = OUT / "scene.png"
    canvas.convert("RGB").save(out)

    big = canvas.resize((canvas.width * SCALE, canvas.height * SCALE), Image.NEAREST)
    big.convert("RGB").save(OUT / "scene_x2.png")

    print(f"scene {canvas.width}x{canvas.height} native -> {out}")
    print(f"      {big.width}x{big.height} at {SCALE}x NEAREST -> scene_x2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
