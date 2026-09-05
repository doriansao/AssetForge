# Findings

Everything here was measured on this stack: a 16GB RTX 4060 Ti, Flux Schnell at
4 steps, SDXL base, SD 1.5, and AudioCraft's medium models. Numbers are from
actual runs, not estimates.

The rejections matter as much as the confirmations. Several of these were
plausible enough that they were recommended before being tested, and then
measured to be neutral or harmful. The reasons are usually specific to this
stack rather than general, so a rejection here is not a claim that the technique
is bad everywhere.

## Confirmed

### The grid snapper produces genuinely indexed art

A shield icon, rendered at 1024px and reduced to 34x48:

| Method | Unique colours |
|---|---|
| Raw Flux render | 64302 |
| Naive Lanczos resize | 1163 |
| Grid snapper | 16 |

1163 colours in 1584 pixels means nearly every pixel is unique, which is what
anti-aliasing does to a downscale. The snapped version has binary alpha and fits
an 8-bit palette losslessly. It is real pixel art; the resized one is a
photograph of pixel art.

### Detect the grid, do not force it

The snapper's target-size control has two modes. `force` drives the grid walker
at the requested cell count directly; `detect` finds the block size the model
actually drew and resamples afterwards.

On a cottage at 48px, `force` lost the window entirely and mangled the chimney.
`detect` kept both. Forcing fights the algorithm's central assumption, because
every cut lands slightly off the real block boundaries.

### Wrap-blending solves tile seams completely

Seam discontinuity, measured as the wrap-point difference divided by the tile's
own average interior step. 1.0 means the wrap is as smooth as normal internal
variation.

| Tile | Raw Flux | After wrap-blend |
|---|---|---|
| grass | 3.18 | 1.11 |
| dirt | 3.54 | 0.89 |
| water | 7.51 | 1.10 |
| stone | 4.23 | 1.09 |
| sand | 5.19 | 0.55 |
| soil | 4.94 | 1.22 |

There is no seam problem left to solve. This is why the circular-padding
experiment below was moot even before its architectural problem was found.

### Fewer palette colours beat more

Three-way comparison of the same tileset and map:

| Palette | Result |
|---|---|
| 128 adaptive | Retains AI-ish smooth gradients in the grass |
| **32 adaptive** | **Best. Flattens texture into readable blocks** |
| DawnBringer 32 (fixed) | Authentic hue-shifted colours, but too few greens; grass speckles badly |

A hand-designed palette carries deliberate hue shifts that an adaptive palette
cannot invent, since adaptive only redistributes colours the render already had.
That is a real advantage, but it is defeated when the palette lacks range in the
dominant hue of your art. DawnBringer 32 would likely win for a subject with a
broader colour spread.

### SDXL plus pixel-art-xl gives the most convincing pixel art

Same chest prompt and seed across four configurations. SDXL with the
`pixel-art-xl` LoRA produced genuinely chunky blocks straight from the model,
where Flux produced a smooth illustration that needed the snapper to become
pixel art. SDXL *without* the LoRA followed the prompt noticeably worse than
Flux, ignoring "a single chest" and producing a sheet of chests.

### The 47-tile blob set is the right count

`autotile.py` enumerates all 256 neighbourhoods, collapses diagonals that cannot
matter, and arrives at exactly 47 distinct tiles. That is the known correct count
for a blob set, which validates the reduction.

## Rejected

### Circular convolution padding for seamless generation

**Inapplicable to Flux.** Tensor ranks in the two model files:

| Model | 1D | 2D (Linear) | 4D (Conv) |
|---|---|---|---|
| Flux denoiser | 464 | 312 | 0 |
| VAE | 174 | 0 | 70 |

The denoiser contains no convolution layers, so there is no padding mode to
change. The technique is real and worth trying on **SDXL and SD 1.5**, which are
UNets and are now available in this repo. It was never going to help Flux.

Moot regardless: wrap-blending already reduces seams to about 1.0.

### Pixel-art LoRAs for cross-asset consistency

Mean pairwise palette distance across a six-object set:

| Condition | Distance |
|---|---|
| No LoRA | 0.622 |
| Top-down pixel art LoRA | 0.618 |

No meaningful change. Worse, Flux pixel-art LoRAs are all trained on FLUX.1-dev
and behave unreliably on schnell: across the same set the barrel became an empty
hoop and the lantern lost its glow.

What actually fixes cross-asset consistency is post-processing. Crop to content,
scale to a fixed box, and snap to one shared palette. The variation that matters
most is scale and framing, and no model setting addresses that.

LoRA support is still wired up, because a LoRA does improve individual renders,
and `pixel-art-xl` on SDXL is genuinely good. It is a per-asset style tool, not a
consistency tool.

### The grid snapper for fixed-size tileset sprites

Under the snapper the barn lost its roof structure and the lamp broke apart,
while a plain box downscale kept both. The cause is double resampling: the
snapper recovers the block grid, then the sprite has to be resized onto a fixed
box size, and that second resize destroys what was just recovered.

`SNAP_SPRITES = False` in `make_tiles.py` records this. The snapper is still the
right tool wherever a sprite may keep its own detected resolution.

### Whole-tile Bayer dithering at terrain borders

At 8px cells it reads as a chessboard, not a blend. Replaced by a shared
noise field that warps each region's edge, which is both subtler and handles
features of any width.

### A seam metric does not transfer from images to audio

The image engine measures a tile seam as the wrap discontinuity divided by the
tile's own mean interior step, where 1.0 means seamless. Reusing that formula for
audio loops reported 3.45 on a track whose loop is continuous *by construction*,
because the cross-fade makes the first and last samples adjacent samples of the
source.

The problem is distribution shape. On that track:

| Statistic | Value |
|---|---|
| Mean sample-to-sample step | 0.015 |
| Median | 0.011 |
| 95th percentile | 0.046 |
| Maximum | 0.224 |
| Wrap step | 0.053 |

A wrap at 3.45x the mean sits at the 97th percentile and well below the maximum,
which is unremarkable. Audio steps are heavily skewed where image gradients
across a tile edge are not. The metric now reports a percentile.

### Contact shadows are a keying problem, not a shape problem

Every diffusion model draws a contact shadow, a dirt disc or a stone base under
an isolated subject regardless of the prompt, and Flux Schnell ignores negative
prompts entirely so it cannot be suppressed at generation time. Left in, it does
two kinds of damage: it dirties the sprite, and because it sits below the feet it
becomes the bottom of the bounding box, so bottom-anchored placement rests the
*shadow* on the ground and the character floats above it.

What works is keying wide enough to absorb it. A shadow on white is a soft grey
that shades continuously out of the background, so a flood fill with a loose
tolerance walks straight into it, while the subject stays outside the threshold
because it is darker and saturated. On one render:

| Flood-fill tolerance | Sprite height | Bottom rows |
|---|---|---|
| 72 | 904px | 0, 0, 1, 2 (shadow tail) |
| 130 | 809px | 48, 48, 49 (flat on the boots) |

The opaque pixel count barely moved, so it removed the shadow and nothing else.
The stable window is roughly 120 to 240; past 300 it starts eating the subject.
The default is now 130.

Two other approaches were tried and are worth knowing about.

**Shape-based trimming, partially kept.** Scanning the alpha for the point where
the subject narrows to its contact point and the plinth widens again does work,
but width alone cannot tell a shadow from a pair of boots: the narrowest row in
the search band is usually the bottom edge of the shadow itself, and searching
further up the sprite finds the gap between a character's legs and cuts the legs
off. Switching the test from width to *brightness* removes that failure mode,
because legs are as dark as the body while a shadow on white is not. It survives
as a second pass for whatever the loose key leaves behind.

**A magenta chroma key, tried and reverted.** It removes shadows perfectly, since
a shadow on the key is a darker shade of the key and a hue test catches it at any
brightness. But the key's strong channels must not be the subject's, and
magenta's are red and blue, which a palette of warm wood and blue-grey stone both
have: the keyer punched holes straight through crates and left magenta fringing
on stone. The implementation is kept for palettes that genuinely suit one.

### Crop before measuring anything about the bottom of a sprite

The brightness trim above silently did nothing for a full day of debugging. It
guards against over-trimming by requiring that most of the rows it is about to
remove look like plinth, and on an uncropped frame the band of empty rows beneath
the subject dilutes that fraction below the guard. Cropping to content first
fixed every case at once.

### A theme phrase must describe palette, not setting

Generating a coherent pack means repeating a shared phrase across every subject.
The choice of phrase matters more than expected.

`"moonlit forest ruins, cool blue and teal palette with warm lantern accents"`
appended to every subject produced, for every sprite, a fully painted forest
scene behind the subject rather than a cutout. The preset's own clause,
"isolated as a cutout on a plain solid pure white background", lost to it. The
background keyer then removed only the white margin around the painted scene,
so every sprite composited as a rectangle of forest, and every 32px terrain tile
was a tiny landscape instead of a texture.

`"cool teal and slate blue palette with warm amber highlights"` produces the same
cohesion without competing with the framing.

The rule: a theme may name colours, materials and mood. The moment it names a
*place*, it becomes a scene instruction and overrides the isolation clause.

## Constraints worth knowing

- **Autotiling needs regions at least 2 tiles thick.** A 1-tile shore breaks into
  disconnected pieces, because both sides are cut back. Verified by widening the
  shore to 2 tiles, which fixed it.
- **AudioGen outputs 16kHz.** Fine for impacts, thuds and ambience; thin for
  anything needing bright high end, like glass or cymbals. Stable Audio Open at
  44.1kHz is the upgrade path, but it is gated on Hugging Face.
- **Flood-fill background keying cannot reach enclosed holes.** The gaps inside a
  fence are ringed by posts and rails, so the fill never gets there. Handled with
  an opt-in flag for sprites with no legitimately white parts.
- **Lanczos before keying breaks the fill.** It rings around dark edges and tints
  thin white gaps just enough to fail the threshold. Use BOX.

## Environment notes

Getting AudioCraft working on Windows took four attempts, recorded here because
the failure modes are not obvious:

1. `pip install audiocraft` fails building `av` from source, because the pinned
   version has no wheel for Python 3.11 on Windows.
2. Installing a newer `av` first does not help; audiocraft still resolves to the
   pinned one.
3. `--no-deps` plus explicit dependencies gets audiocraft installed, but it
   imports `xformers` unconditionally.
4. xformers has no wheel for torch 2.5.1 on this platform, and building from
   source fails on Windows path length limits inside its vendored
   flash-attention tree.

The combination that works is **torch 2.4.0 + torchaudio 2.4.0 +
xformers 0.0.27.post2**, all from the cu121 index, which have matching prebuilt
Windows wheels. This is why the audio engine has its own virtualenv.
