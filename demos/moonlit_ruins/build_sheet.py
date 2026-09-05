#!/usr/bin/env python3
"""Contact sheet for the Moonlit Ruins pack.

Shows every asset at an integer NEAREST zoom with its true pixel dimensions, so
the sheet is a readable inventory rather than a mood board. Terrain is shown
tiled 3x3 to make a repeat failure obvious.

  python demos/moonlit_ruins/build_sheet.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
PACK = ROOT / "out" / "image" / "moonlit_ruins"

BG = (26, 28, 38)
FG = (225, 229, 238)
DIM = (140, 148, 166)

# name, label, tile it to prove the repeat
ITEMS = [
    ("ground", "ground", True),
    ("underground", "underground", True),
    ("crate", "crate", False),
    ("brazier", "brazier", False),
    ("pillar", "pillar", False),
    ("hero", "hero", False),
    ("enemy", "golem", False),
    ("icon_potion", "potion", False),
    ("icon_key", "key", False),
    ("icon_gem", "gem", False),
]

CELL_W, CELL_H = 168, 190


def cell(name: str, label: str, tiled: bool) -> Image.Image:
    img = Image.open(PACK / f"{name}.png").convert("RGBA")
    native = img.size

    if tiled:
        t = Image.new("RGBA", (img.width * 3, img.height * 3))
        for y in range(3):
            for x in range(3):
                t.paste(img, (x * img.width, y * img.height))
        img = t

    zoom = max(1, min((CELL_W - 24) // img.width, (CELL_H - 52) // img.height))
    shown = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)

    panel = Image.new("RGBA", (CELL_W, CELL_H), BG)
    panel.alpha_composite(shown, ((CELL_W - shown.width) // 2,
                                  14 + (CELL_H - 52 - shown.height) // 2))
    d = ImageDraw.Draw(panel)
    d.text((10, CELL_H - 32), label, fill=FG)
    suffix = "  (tiled 3x3)" if tiled else ""
    d.text((10, CELL_H - 18), f"{native[0]}x{native[1]}px  {zoom}x{suffix}", fill=DIM)
    return panel


def main() -> int:
    cols = 5
    rows = (len(ITEMS) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * CELL_W + 16, rows * CELL_H + 46), BG)
    d = ImageDraw.Draw(sheet)
    d.text((12, 12), "MOONLIT RUINS  -  image pack", fill=FG)
    d.text((12, 28), "every asset generated and post-processed by AssetForge", fill=DIM)

    for i, (name, label, tiled) in enumerate(ITEMS):
        sheet.alpha_composite(cell(name, label, tiled),
                              (8 + (i % cols) * CELL_W, 46 + (i // cols) * CELL_H))

    out = PACK / "pack_sheet.png"
    sheet.convert("RGB").save(out)
    print(f"sheet {sheet.width}x{sheet.height} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
