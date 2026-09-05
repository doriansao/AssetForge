# Recipes

End-to-end workflows. Each assumes the ComfyUI server is running for image work
(see [SETUP.md](SETUP.md)) and uses the audio venv interpreter for audio work.

Paths below are relative to the repository root.

## A starter asset pack for a platformer

Six images and six sounds, enough to prototype a level.

```bash
IMG=engines/image/scripts
AUD=./engines/audio/venv/Scripts/python.exe

python $IMG/asset_presets.py --preset platformer_tile --subject "cracked mossy stone brickwork" --output out/image/pack/ground.png
python $IMG/asset_presets.py --preset platformer_prop --subject "a wooden crate bound with iron bands" --output out/image/pack/crate.png
python $IMG/asset_presets.py --preset character --subject "a hooded rogue with twin daggers" --output out/image/pack/hero.png --normal-map --keep-raw
python $IMG/asset_presets.py --preset enemy --subject "a squat armoured goblin with a club" --output out/image/pack/goblin.png
python $IMG/asset_presets.py --preset item_icon --subject "a red health potion" --output out/image/pack/potion.png
python $IMG/asset_presets.py --preset parallax_layer --subject "distant blue mountains under a pale sky" --output out/image/pack/bg_far.png

$AUD engines/audio/scripts/audio_presets.py --preset jump --output out/audio/pack/jump.wav
$AUD engines/audio/scripts/audio_presets.py --preset land --output out/audio/pack/land.wav
$AUD engines/audio/scripts/audio_presets.py --preset sword_hit --output out/audio/pack/hit.wav
$AUD engines/audio/scripts/audio_presets.py --preset pickup_coin --output out/audio/pack/coin.wav
$AUD engines/audio/scripts/audio_presets.py --preset footstep_stone --output out/audio/pack/step.wav --count 4
$AUD engines/audio/scripts/audio_presets.py --preset battle_theme --output out/audio/pack/battle.wav
```

The `--count 4` on footsteps is deliberate. Shipping four variations and picking
randomly at runtime is how games avoid the machine-gun effect of one repeated
sample.

Keep `--era` consistent across every image call, or the pack will not look like
one set. If you also append a shared theme phrase for cohesion, **make it
describe palette and material only**. A phrase naming a place, such as "moonlit
forest ruins", overrides the preset's isolation clause: the model paints that
scene behind every sprite, keying then strips only the white margin, and each
sprite composites as a rectangle. "cool teal and slate blue palette with warm
amber highlights" gives the same cohesion without the fight.

## A tileset and a map

```bash
python engines/image/scripts/gen_tileset.py            # ~20 renders, a few minutes
python engines/image/scripts/make_tiles.py             # -> out/image/tiles/
python engines/image/scripts/build_village.py          # -> out/image/village_map.png
```

To change one tile without re-rendering the rest, pass its name:

```bash
python engines/image/scripts/gen_tileset.py water stone
python engines/image/scripts/make_tiles.py
```

Edit the prompt tables at the top of `gen_tileset.py` to change what gets made.
Tile names there must match the terrain and object names used in
`build_village.py` and `make_tiles.py`.

## Exporting terrain transitions for Godot or Unity

```bash
python engines/image/scripts/autotile.py --base grass --over dirt
python engines/image/scripts/autotile.py --base grass --over water
python engines/image/scripts/autotile.py --base grass --over sand
```

Each writes `<over>_on_<base>_atlas.png` and a matching `.json` to
`out/image/autotiles/`. The JSON maps a neighbourhood bitmask to a tile position
in the atlas, with the bit order documented in the file.

Remember the constraint: **regions must be at least 2 tiles thick**. A one-tile
shore breaks into disconnected pieces. For thin features use
`build_village.py --borders noise` instead.

## Lighting a tileset

```bash
python engines/image/scripts/make_normalmap.py -i out/image/tiles -o out/image/normals
```

One `<name>_n.png` per tile. In Godot, assign it to a CanvasTexture's normal map
slot; in Unity, as the secondary texture on the Sprite Renderer with a 2D light
in the scene. Raise `--strength` for more pronounced relief, lower `--detail`
towards 0 if surface texture is reading as dents rather than as texture.

## Reprocessing without regenerating

Post-processing is cheap and generation is not, so iterate on the raw frame
rather than re-rendering. `generate_asset.py --pixel-art` always keeps it as
`<output>_raw.png`; `asset_presets.py` needs `--keep-raw`, which is why the
character line in the pack recipe above passes it.

```bash
python engines/image/scripts/pixel_art_processor.py -i out/image/pack/hero_raw.png \
    -o out/image/pack/hero_32.png --target-size 32 --palette-size 16 --upscale-factor 8
```

The `--upscale-factor` preview is for looking at, not for shipping. Ship the
small file and let the engine scale it with nearest-neighbour filtering.

## Trying a different look

Same subject and seed, three models:

```bash
P="pixel art, a wooden treasure chest with iron bands, game item sprite, plain solid white background"
python engines/image/scripts/generate_asset.py --prompt "$P" --output out/image/try/flux.png --seed 7
python engines/image/scripts/generate_asset.py --model sdxl --lora pixel-art-xl.safetensors --lora-strength 1.2 --prompt "$P" --output out/image/try/sdxl.png --seed 7
python engines/image/scripts/generate_asset.py --model sd15 --prompt "$P" --output out/image/try/sd15.png --seed 7
```

Flux follows the prompt best. SDXL with `pixel-art-xl` gives the most convincing
pixel blocks. SDXL without a LoRA is the weakest at prompt compliance and will
sometimes hand back a sheet of variations instead of one object.

## Adding a preset

**Image**, in `engines/image/scripts/asset_presets.py`. Add an entry to
`PRESETS` with `framing` (a prompt template using `{s}` for the subject), `size`
(render resolution), `post` (`snap`, `box` or `tile`), `target`, and `palette`.

The `post` choice is the one that matters: use `snap` when the sprite may keep
its own detected resolution, `box` when it must land on an exact size, and
`tile` for anything that has to repeat.

**Audio**, in `engines/audio/scripts/audio_presets.py`. Add an entry to
`PRESETS` as `(kind, duration_seconds, prompt)`. Write the prompt in terms of
material, impact, decay and space rather than naming the thing, and keep the
duration tight.

## Prompting notes that generalise

- Say what the object *is made of* and *how it is lit*, not just what it is.
- For sprites, always include an isolating clause: "isolated as a cutout on a
  plain solid pure white background, no shadow". Models add shadows anyway; for
  SDXL and SD 1.5 the default negative prompt suppresses them, and Flux ignores
  negatives entirely because it is guidance-distilled.
- For audio, "close and dry" suppresses reverb. You generally want that, because
  reverb is better added by the engine per-space than baked into the sample.
