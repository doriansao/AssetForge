# Models

Which model to use, why, and what else is worth trying. Measured comparisons are
in [FINDINGS.md](FINDINGS.md).

## Image models

Selected with `--model` on `generate_asset.py` and `asset_presets.py`. Defined in
`engines/image/models.py`; each has a matching ComfyUI API workflow in
`engines/image/workflows/`.

| Model | Native | Steps | Speed | Best at |
|---|---|---|---|---|
| `flux-schnell` (default) | 1024px | 4 | ~5s | Prompt adherence, framing |
| `sdxl` | 1024px | 25 | ~13s | Pixel art via LoRA, huge LoRA ecosystem |
| `sd15` | 512px | 25 | ~6s | Speed, ControlNet pose control |

**flux-schnell** follows prompts better than either alternative. It respects
"a single X isolated on a white background" where SDXL will hand you a sheet of
variations. It is guidance-distilled, so it runs at cfg 1.0 and ignores negative
prompts entirely. Its weakness is the LoRA ecosystem: thin, and trained for
FLUX.1-dev rather than schnell, which behaves unreliably here.

**sdxl** is the one to use when the pixel look matters more than exact prompt
compliance. With `pixel-art-xl` at strength 1.2 it produces genuinely chunky
blocks straight from the model, rather than a smooth illustration that needs the
grid snapper afterwards. Without a LoRA its prompt adherence is the weakest of
the three. It is a UNet, so ControlNet and circular-padding tricks apply.

**sd15** is small and fast with the deepest ControlNet ecosystem, which is what
you want for pose-controlled character sprite sheets. Prompt adherence is the
weakest; expect to iterate.

### LoRAs

Drop `.safetensors` files into `engines/image/ComfyUI/models/loras/` and pass
`--lora <filename> --lora-strength <n>`. The loader is spliced in front of both
the denoiser and the text encoder, because a style LoRA changes how its trigger
word is embedded, not only how it is denoised.

| LoRA | Base | Trigger | Notes |
|---|---|---|---|
| `pixel-art-xl` | SDXL | "pixel art" | The reference pixel-art LoRA. Use strength ~1.2. |
| `top-down-pixel-art-flux-lora` | Flux dev | none | Works on schnell, improves framing. |
| `ume_modern_pixelart` | Flux dev | `umempart` | Produced artefacts on schnell at 0.9. |

### Worth trying, not wired up

- **SDXL-Turbo / SDXL-Lightning** — 1 to 4 step SDXL. Would make SDXL as fast as
  Flux, at some quality cost. The obvious next addition.
- **ControlNet for SD 1.5 or SDXL** — the real unlock for character work: pose a
  stick figure, get the same character in every animation frame. This is the
  strongest remaining lead for sprite-sheet consistency.
- **IP-Adapter** — style transfer from a reference image. Another route at
  cross-asset consistency, which a LoRA measurably failed to deliver.
- **Qwen-Image, PixArt-Sigma** — newer open models with different strengths;
  untested here.

## Audio models

Defined in `engines/audio/scripts/models.py`. Both families are Meta AudioCraft
and share one API. Select with `--kind music|sfx` or `--model`.

| Model | Kind | Params | VRAM | Rate |
|---|---|---|---|---|
| `musicgen-small` | music | 300M | ~2GB | 32kHz |
| `musicgen-medium` (default) | music | 1.5B | ~6GB | 32kHz |
| `musicgen-large` | music | 3.3B | ~12GB | 32kHz |
| `musicgen-melody` | music | 1.5B | ~6GB | 32kHz |
| `audiogen-medium` (default) | sfx | 1.5B | ~6GB | 16kHz |

They are not interchangeable. MusicGen was trained on music and produces tonal,
rhythmic material; AudioGen was trained on environmental sound. Asking MusicGen
for a sword hit yields a short piece of music that vaguely resembles one.

**musicgen-melody** deserves attention for game soundtracks specifically: it
accepts a reference melody alongside text, so you can generate variations of one
theme for different areas. That is the cheapest way to make a soundtrack feel
composed rather than assembled from unrelated clips.

**audiogen-medium**'s 16kHz output is its main limit. Impacts, thuds, footsteps
and ambience are fine. Anything needing bright high end, like glass breaking or
cymbals, comes out thin.

### Worth trying, not wired up

- **Stable Audio Open 1.0** (`stabilityai/stable-audio-open-1.0`) — the best open
  model for sound effects: 44.1kHz stereo, up to 47 seconds, far brighter than
  AudioGen. Gated on Hugging Face, so it needs an accepted licence and a token,
  and it runs through `stable-audio-tools` rather than AudioCraft. This is the
  upgrade path when AudioGen's 16kHz ceiling bites.
- **AudioLDM 2** (`cvssp/audioldm2`) — ungated text-to-audio through diffusers,
  16kHz. Similar quality band to AudioGen with an easier install; useful as a
  second opinion on a prompt.
- **Bark** (`suno/bark`) — speech and non-verbal vocal sounds. The practical
  choice for creature vocalisations and grunts, which neither MusicGen nor
  AudioGen handles well.

## Licences

These differ, and it matters if you ship.

| Model | Licence | Commercial use |
|---|---|---|
| FLUX.1-schnell | Apache 2.0 | Yes |
| SDXL 1.0 | CreativeML Open RAIL++-M | Yes, with use restrictions |
| SD 1.5 | CreativeML Open RAIL-M | Yes, with use restrictions |
| MusicGen / AudioGen weights | **CC-BY-NC 4.0** | **No** |
| Stable Audio Open | Stability Community Licence | Free under a revenue threshold |

AudioCraft's weights being non-commercial is the one most likely to catch you
out. The code is MIT; the weights are not. If you are shipping a commercial
game, treat MusicGen and AudioGen output as prototyping material and check
Stable Audio Open or a licensed library for final assets.
