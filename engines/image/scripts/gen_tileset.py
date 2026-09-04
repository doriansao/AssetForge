#!/usr/bin/env python3
"""Generate the raw Flux renders that back the tileset.

Each entry produces one high-resolution source image under assets/raw/. These are
not the final tiles -- make_tiles.py downsamples and quantises them into true
pixel art. Seeds are fixed so a re-run reproduces the same village.

Terrain sources are flat repeating textures; object sources are single sprites on
a plain white field so the background can be keyed out later.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_asset import build_workflow, download, first_image, poll, submit, wait_for_server

import paths

RAW = str(paths.RAW)

STYLE = "pixel art, 16-bit SNES era JRPG tileset, Stardew Valley art style, crisp clean pixels, flat even lighting, vibrant saturated palette"

# name, prompt fragment, size, seed
TERRAIN = [
    ("grass",   "top-down seamless texture of lush green meadow grass with subtle blade detail", 512, 1001),
    ("grass2",  "top-down seamless texture of meadow grass scattered with tiny yellow and white wildflowers", 512, 1201),
    ("grass3",  "top-down seamless texture of short green lawn grass with darker tufts and clover patches", 512, 1202),
    ("dirt",    "top-down seamless texture of a walking path of packed earth, muted chocolate brown and tan, scattered small grey pebbles, desaturated natural soil colour, not orange, not red", 512, 1102),
    ("water",   "top-down seamless texture of pond water seen from directly above, medium cobalt blue with bold chunky pale blue wave lines and dark navy depth patches, high contrast, large clearly defined ripple shapes", 512, 1303),
    ("stone",   "top-down seamless texture of irregular rounded cobblestones of varying size fitted together, weathered grey granite, uneven organic layout", 512, 1104),
    ("sand",    "top-down seamless texture of pale golden beach sand", 512, 1005),
    ("soil",    "top-down seamless texture of freshly ploughed crumbly farm earth, dark brown loose soil with soft parallel furrow ridges, matte and damp, not bricks, not tiles", 512, 1106),
]

OBJECTS = [
    ("tree_oak",   "a single lush round oak tree with thick green canopy, seen from above at a slight angle", 768, 2001),
    ("tree_pine",  "a single broad conical evergreen pine tree with dense full layered branches reaching wide at the base, seen from above at a slight angle", 768, 2102),
    ("house_red",  "a small cosy cottage with a red pitched roof, stone chimney and wooden door, three quarter view", 768, 2003),
    ("house_blue", "a small cosy cottage with a blue pitched roof, white window frames and a wooden door, three quarter view", 768, 2004),
    ("barn",       "a wooden farm barn with a tall gambrel roof and big double doors, three quarter view", 768, 2005),
    ("well",       "a round stone water well with a small wooden roof and bucket", 768, 2006),
    ("bush",       "a small round leafy bush covered in pink and yellow flowers", 768, 2007),
    ("rock",       "a single grey mossy boulder rock", 768, 2008),
    ("fence",      "one section of sturdy wooden ranch fence with two thick horizontal rails and square posts, weathered pale brown timber, side view, spanning the full width", 768, 2209),
    ("crop",       "a leafy green cabbage crop plant growing in a mound of soil, seen from above", 768, 2010),
    ("lamp",       "a wrought iron street lamp post with a warm glowing lantern on top, side view", 768, 2011),
    ("haystack",   "a stack of rectangular golden straw hay bales piled two high on a farm, visible straw texture and binding twine", 768, 2212),
]


def render(name, prompt, size, seed, force=()):
    out = os.path.join(RAW, f"{name}.png")
    if name in force:
        pass
    elif os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[skip] {name}", flush=True)
        return
    wf = build_workflow(prompt, size, size, seed)
    started = time.time()
    entry = poll(submit(wf))
    download(first_image(entry), out)
    print(f"[done] {name:11s} {size}px  {time.time()-started:5.1f}s", flush=True)


def main():
    os.makedirs(RAW, exist_ok=True)
    wait_for_server()
    force = set(sys.argv[1:])

    for name, desc, size, seed in TERRAIN:
        render(name, f"{desc}, {STYLE}, no objects, no shadows, fills the entire frame", size, seed, force)

    for name, desc, size, seed in OBJECTS:
        render(name, f"{desc}, {STYLE}, single game sprite centered in frame, "
                     f"isolated as a cutout on a plain solid pure white background, "
                     f"no shadow, no ground patch, no soil, no grass base, no platform, no text",
               size, seed, force)

    print("\nraw sources written to", RAW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
