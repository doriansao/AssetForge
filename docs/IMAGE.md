# Image engine

Drives ComfyUI headlessly over its REST API, then post-processes renders into
pixel art. See [MODELS.md](MODELS.md) for model choice and
[FINDINGS.md](FINDINGS.md) for measurements.

## Running the server

```bash
cd engines/image && ./python_embeded/python.exe -s ComfyUI/main.py --windows-standalone-build --listen 127.0.0.1 --port 8188
```

Every generation script waits for `http://127.0.0.1:8188/system_stats` before
submitting, so start the server first. First render after a restart pays a model
load of roughly 13 seconds; subsequent ones are about 5.

Models are fetched by `scripts/download_models.py`, which skips files already
present at the right size.

## Presets

`scripts/asset_presets.py` is the entry point for a single asset. A preset pins
the three things that differ between asset types: framing, seamlessness, and
resolution with its matching post-processor.

```bash
python engines/image/scripts/asset_presets.py --list
python engines/image/scripts/asset_presets.py --preset character \
    --subject "a hooded rogue with twin daggers" --output out/image/rogue.png --normal-map
```

| Preset | Output | Post | For |
|---|---|---|---|
| `topdown_tile` | 32px, seamless | tile | Top-down terrain |
| `topdown_object` | 64px | box | Trees, buildings, props |
| `platformer_tile` | 32px, seamless | tile | Side-on ground and walls |
| `platformer_prop` | 48px | box | Crates, barrels, spikes |
| `character` | native | snap | Side-on player character |
| `enemy` | native | snap | Side-on enemy |
| `character_portrait` | native | snap | Dialogue portrait |
| `item_icon` | 32px | snap | Inventory and UI |
| `parallax_layer` | native, h-seamless | tile | Scrolling backgrounds |

`--era` picks the visual vocabulary: `8bit`, `16bit` (default), `gba`, `modern`.

## Post-processing

`scripts/pixel_art_processor.py`, usable standalone or as a library.

The core is a grid snapper vendored from
[doriansao/codex-pixel-art-generator](https://github.com/doriansao/codex-pixel-art-generator)
(MIT), itself a port of Hugo Duprez's spritefusion-pixel-snapper (MIT). It works
in five steps:

1. k-means++ quantise at full resolution, so outline pixels get their own colour
   centroid before any spatial decision
2. build 1D gradient profiles per axis
3. estimate the native block size from median peak-to-peak distance
4. walk each profile in floating point, snapping cuts to gradient peaks
5. take the per-cell mode of RGBA, so a one-pixel outline wins its cell instead
   of being averaged away

Added on top, because the upstream skill gets these from elsewhere in its own
workflow: background keying to alpha, content cropping, a target-size bridge,
and integer NEAREST upscaling.

```bash
python engines/image/scripts/pixel_art_processor.py -i render.png -o sprite.png \
    --target-size 48 --palette-size 16 --upscale-factor 8
```

**Snap or box** is the decision that matters. Snapping recovers a grid; resizing
afterwards destroys it. Use the snapper when a sprite may keep its own detected
resolution, and a box downscale when it must land on an exact size. Measured
evidence is in FINDINGS.md.

Palette size by subject: 8 to 16 for props and constrained eras, 24 for detailed
props, 32 to 48 for anything with human skin, which needs base, shadow and
highlight tones to stay separate.

## Keying and reference edits

Two scripts beside `generate_asset.py` cover what a text-to-image render cannot:
a clean alpha, and the same object from another angle.

### remove_background.py (BiRefNet)

```bash
python engines/image/scripts/remove_background.py --input render.png --output sprite.png --crop
```

Drives ComfyUI's native `RemoveBackground` node with
`models/background_removal/birefnet.safetensors` (from `Comfy-Org/BiRefNet`).
BiRefNet is a matting model, so it keeps near-white foreground on a white
plate and gives soft anti-aliased edges. On the Cowduction set it cut five
renders cleanly in about a second each, including an ivory hull on white and
a pixel-art render, where the flood-fill keyer in `pixel_art_processor.py`
had eaten the hull panels. Use BiRefNet for smooth high-resolution sprites;
keep the flood fill for true pixel art, where a soft edge is wrong.

### edit_asset.py (Qwen-Image-Edit-2511, FLUX.2 klein 4B)

```bash
# same object, new camera (fal's Multiple Angles LoRA)
python engines/image/scripts/edit_asset.py --model qwen-edit --ref hull.png \
    --angle "front view high-angle shot medium shot" --output hull_top.png
# instruction edit
python engines/image/scripts/edit_asset.py --model qwen-edit --ref hull.png \
    --prompt "Open the cargo hatch on the top deck. Keep everything else identical." --output hatch.png
# klein: text to image, reference edit, or a 2x2 multi-view sheet
python engines/image/scripts/edit_asset.py --model klein --prompt "..." --output out.png
python engines/image/scripts/edit_asset.py --model klein --ref hull.png --prompt "the same saucer from directly above" --output top.png
python engines/image/scripts/edit_asset.py --model klein --ref turret.png --lora flux-2-klein-4b-spritesheet-lora.safetensors --prompt "2x2 sprite sheet" --output sheet.png
```

`qwen-edit` runs Qwen-Image-Edit-2511 as a Q4_K_M GGUF (12.3 GB, needs the
`ComfyUI-GGUF` custom node) with the 4-step Lightning LoRA, `qwen_2.5_vl_7b`
fp8 as text encoder and the Qwen-Image VAE. `--angle` adds the Multiple Angles
LoRA and prefixes its `<sks>` trigger; the vocabulary is in the script's
docstring. About 90 s per edit on a 16GB card, most of it model shuffling.

`klein` runs FLUX.2 klein 4B distilled (bf16, 7.2 GB) with the Qwen3-4B text
encoder and the FLUX.2 VAE through `Flux2Scheduler` + `SamplerCustomAdvanced`,
4 steps, cfg 1, 9 to 17 s per image. References are fed as `ReferenceLatent`
conditioning, one per `--ref`. fal's sprite-sheet LoRA turns one object into a
2x2 sheet: two isometric views, a side view and a top-down view.

All weights are Apache 2.0 and are downloaded by hand from Hugging Face; see
`docs/MODELS.md` for file names and folders.

**VRAM trap.** The Q4 GGUF loads "completely" at 12.7 GB and leaves nothing
for the text encoder and activations, and the run sits at step 0 forever at
100 % GPU. Start the server with `--reserve-vram 2.5` so ComfyUI offloads part
of the model to pinned RAM, and make sure no other server (the mesh engine on
port 8189 keeps 6 GB resident after a job) is holding the card.

## Tilesets

Three stages, deliberately separate so you can re-run one without the others.

```bash
python engines/image/scripts/gen_tileset.py                 # render sources
python engines/image/scripts/make_tiles.py                  # -> tiles + sprites
python engines/image/scripts/build_village.py               # -> composed map
```

`gen_tileset.py` accepts tile names to re-render selectively, for example
`gen_tileset.py dirt water`, which overwrites only those and leaves the rest.

`make_tiles.py` makes terrain seamless by cross-fading each edge into its
opposite, downsamples to 32px, and snaps everything to one shared adaptive
palette. The shared palette is what makes 20 independently generated tiles look
like one artist drew them. `--palette` accepts `adaptive` (32 colours, the
default), `adaptive128`, or `db32` for a fixed DawnBringer 32. Fewer colours
measured better here; see [FINDINGS.md](FINDINGS.md).

`build_village.py --borders` chooses how terrain regions meet:

- `noise` (default) warps each region's edge with a shared noise field. Organic,
  handles features of any width.
- `autotile` uses the 47-tile blob set. Crisp, engine-standard, but needs regions
  at least 2 tiles thick.

## Autotiling

```bash
python engines/image/scripts/autotile.py --base grass --over dirt
```

Writes an atlas PNG and a JSON lookup that Godot 4 terrain sets or Unity rule
tiles can consume. Each tile is assembled from four quadrants, and a quadrant has
only five possible shapes, which reproduces the full 47-tile set exactly while
guaranteeing neighbouring tiles stay consistent.

## Normal maps

```bash
python engines/image/scripts/make_normalmap.py -i out/image/tiles -o out/image/normals
```

Height is inferred from luminance for surface detail, blended with an alpha
falloff that rounds the sprite off at its outline. Luminance alone reads a black
outline as a trench, so the silhouette term is what keeps sprites reading as
solid objects. Output uses the OpenGL convention that Godot and Unity expect.

## Adding a model

1. Put the checkpoint in `engines/image/ComfyUI/models/checkpoints/`.
2. Add a workflow JSON to `engines/image/workflows/`, keeping the node ids
   consistent: 4 positive, 5 negative, 6 latent, 7 sampler.
3. Add an entry to `engines/image/models.py` with its loader outputs and sampler
   defaults.

Only the loader section differs between workflows, which is what `model_out` and
`clip_out` in the registry describe. Everything else stays model-agnostic.
