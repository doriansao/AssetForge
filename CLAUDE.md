# AssetForge — working notes for an agent

Local 2D game asset generation. Two engines, `engines/image` and `engines/audio`,
with separate Python environments because AudioCraft pins an older torch than
the image engine uses. Never install audio dependencies into the image engine or
the reverse.

Full detail lives in `docs/`. This file is the decision layer: what to reach for,
and the traps that cost time here.

## Before anything else

The image engine needs ComfyUI running:

```bash
cd engines/image && ./python_embeded/python.exe -s ComfyUI/main.py --windows-standalone-build --listen 127.0.0.1 --port 8188
```

Confirm with `curl -s http://127.0.0.1:8188/system_stats`. First render after a
restart pays a model load, roughly 13s; subsequent renders are about 5s.

The audio engine needs no server, but must be run with its own interpreter:
`./engines/audio/venv/Scripts/python.exe`. Running it with the system Python
fails on import.

## Which tool

| You want | Use |
|---|---|
| One image asset, any genre | `engines/image/scripts/asset_presets.py --preset <name>` |
| One sound or music track | `engines/audio/scripts/audio_presets.py --preset <name>` |
| Full control over a render | `engines/image/scripts/generate_asset.py` |
| A tileset | `gen_tileset.py` then `make_tiles.py` |
| A composed map | `build_village.py` |
| Terrain transitions for an engine | `autotile.py` |
| 2D dynamic lighting | `make_normalmap.py` |
| Post-process an existing image | `pixel_art_processor.py` |
| Post-process an existing sound | `audio_processor.py` |

Both preset scripts take `--list`.

## Picking an image model

`--model` on `generate_asset.py` and `asset_presets.py`.

- **flux-schnell** (default) has the best prompt adherence of the three and is
  fast at 4 steps. Use it when the prompt is specific and the framing matters.
- **sdxl** with the `pixel-art-xl` LoRA produces the most convincing pixel art
  straight out of the model, chunky blocks rather than a smooth picture of pixel
  art. Use it when the pixel look matters more than exact prompt compliance.
  SDXL alone, without the LoRA, follows prompts noticeably worse than Flux: it
  ignored "a single chest" and produced a sheet of chests.
- **sd15** is fastest and smallest, weakest at prompts, and has the deepest
  ControlNet ecosystem. Reach for it when you need pose control.

SDXL and SD 1.5 are UNets, so circular-padding seamless generation is available
to them. Flux is a transformer and it is not. See `docs/FINDINGS.md`.

## The decision that matters most: snap or box

Two ways to turn a full-resolution render into a sprite, and choosing wrong is
the usual cause of mushy or broken output.

**Grid snapper** (`pixel_art_processor.py`, `post="snap"`) finds the block grid
the model actually drew and takes one colour per block. Diffusion models do not
draw pixel art, they draw a picture *of* it, so the blocks drift off any fixed
grid and have soft edges. Use it when **the sprite may keep its own detected
resolution**: icons, characters, enemies, portraits, anything standalone.

**Box downscale** (`post="box"`) is a plain area average plus a shared palette.
Use it when **the sprite must land on an exact size**, such as a tileset where
every prop has a fixed footprint.

The rule underneath: snapping recovers a grid, and any resize afterwards
destroys the grid it just recovered. If a resize is unavoidable, do not snap.

Also: prefer `--grid detect` over `--grid force`, and set the palette by subject.
Use 8 to 16 colours for props, 24 for detailed props, and 32 to 48 for anything
with human skin, which needs separate base, shadow and highlight tones.

## Audio conventions

- Effects are short and must **not** loop. Looping cross-fades the tail over the
  attack, which destroys the transient that makes an impact read as an impact.
- Music loops by default, using an equal-power cross-fade of tail over head.
  This is the same trick the image engine uses to make a texture tile.
- Everything is trimmed, resampled to 44.1kHz and peak-normalised to -1 dBFS.
  Trimming matters more than it sounds: leading silence on an effect is input
  lag the player feels.
- Prompt with **material, impact, decay and space**, not with the name of the
  thing. "A steel blade striking a wooden shield, sharp crack with a short woody
  decay, close and dry" works; "sword hit" does not.

## Traps

- **AudioCraft weights are CC-BY-NC 4.0, non-commercial.** Model licences differ
  across this repo; check before shipping generated output.
- **Autotiling needs regions at least 2 tiles thick.** A 1-tile shore breaks into
  disconnected pieces. Use `--borders noise` for thin features.
- **Flood-fill background keying cannot reach enclosed holes**, such as the gaps
  inside a fence. Set `drop_enclosed` for sprites with no legitimately white parts.
- **Resize with BOX, not LANCZOS, before keying.** Lanczos rings around dark
  edges and tints thin white gaps enough to defeat the fill threshold.
- **Never upscale pixel art with anything but NEAREST**, and only by integer
  factors.
- **Black Forest Labs' FLUX.1-schnell repo is gated.** The UNET and VAE come from
  public mirrors; see `download_models.py`.
- **Paths are derived from `paths.py` in each engine.** Do not hard-code absolute
  paths; the repo must stay clonable anywhere.

## Read before changing anything

`docs/FINDINGS.md` records what was measured, including five plausible
improvements that were tested and found neutral or harmful. Among them: circular
padding is inapplicable to Flux, pixel-art LoRAs do not improve cross-asset
consistency, and the grid snapper is worse than a box downscale for fixed-size
tileset sprites. Each has numbers attached. Do not redo them.
