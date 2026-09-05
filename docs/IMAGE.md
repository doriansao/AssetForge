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
