# Moonlit Ruins

A complete vertical-slice asset pack for a 2D platformer, generated end to end
by this repository. Eleven images and ten audio files, plus a composed scene
built from them.

Nothing here was drawn, recorded or edited by hand. The only hand-written parts
are the prompts and the scene layout.

![scene](../../docs/images/moonlit_ruins_scene.png)

## What is in it

**Images**, at `out/image/moonlit_ruins/`

| Asset | Preset | Size |
|---|---|---|
| ground, underground | `platformer_tile` | 32x32, seamless |
| crate, brazier, pillar | `platformer_prop` | ~48px |
| hero | `character` | 64px, plus a normal map |
| golem | `enemy` | 64px, plus a normal map |
| potion, key, gem | `item_icon` | 32px |
| bg_far | `parallax_layer` | horizontally seamless |

**Audio**, at `out/audio/moonlit_ruins/`

| Asset | Preset | Length |
|---|---|---|
| jump, land, sword_hit, pickup_gem, golem_break | various sfx | 0.5 to 2s |
| footstep x3 | `footstep_stone --count 3` | ~0.5s each |
| explore_loop | `dungeon_ambient` | 19.6s, seamless loop |
| battle_loop | `battle_theme` | 19.6s, seamless loop |

Three footstep variations rather than one, because a single repeated sample
machine-guns as soon as the player walks.

## Rebuilding it

Generation prompts live in the pack scripts; the two scripts here only compose
what already exists.

```bash
python demos/moonlit_ruins/build_sheet.py    # contact sheet of every asset
python demos/moonlit_ruins/build_scene.py    # the composed level screenshot
```

The scene is laid out at 640x384 native on a 32px grid, then upscaled 2x with
NEAREST. Composing at the final size instead would put sprites on half-pixels
and the result would read as blurry.

## Two things this demo exposed

**A theme phrase must describe palette, not setting.** The first attempt
appended "moonlit forest ruins, cool blue and teal palette with warm lantern
accents" to every subject. The model treated the place-name as a scene
instruction, painted a forest behind every sprite, and the background keyer
removed only the white margin around it, so each sprite composited as a
rectangle of forest. Every 32px terrain tile came out as a tiny landscape rather
than a texture. Replacing it with "cool teal and slate blue palette with warm
amber highlights" fixed the whole pack in one re-run.

**Characters need resizing, and the raw frames make that free.** The `character`
and `enemy` presets keep their own detected resolution, which came out around
136px, over four tiles tall on a 32px grid. Because every render kept its
full-resolution frame, dropping them to 64px was a post-processing pass rather
than a re-generation:

```bash
python engines/image/scripts/pixel_art_processor.py \
    -i out/image/moonlit_ruins/hero_raw.png \
    -o out/image/moonlit_ruins/hero.png --target-size 64 --palette-size 32
```

Both are recorded in [docs/FINDINGS.md](../../docs/FINDINGS.md).
