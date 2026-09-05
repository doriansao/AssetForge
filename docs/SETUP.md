# Setup

From a fresh clone to a working install. The two engines are set up
independently and neither depends on the other at runtime.

Everything here was performed on Windows 11 with an RTX 4060 Ti (16GB). Budget
roughly 40GB of disk for models and about an hour, most of it downloading.

## 1. Image engine

### ComfyUI

The image engine drives a ComfyUI portable install that lives at
`engines/image/`. It is not in git.

Download the current **NVIDIA portable** release from
[Comfy-Org/ComfyUI releases](https://github.com/Comfy-Org/ComfyUI/releases).
Note it ships as `.7z`, not `.zip`:

```bash
curl -L -o comfy.7z https://github.com/Comfy-Org/ComfyUI/releases/download/v0.34.0/ComfyUI_windows_portable_nvidia.7z
7z x comfy.7z -o_extract
```

Then move the *contents* of `_extract/ComfyUI_windows_portable/` up into
`engines/image/`, so the layout is:

```
engines/image/ComfyUI/
engines/image/python_embeded/
engines/image/run_nvidia_gpu.bat
```

`engines/image/scripts/paths.py` expects exactly this. Verify the runtime:

```bash
cd engines/image && ./python_embeded/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Model weights

```bash
python engines/image/scripts/download_models.py --list      # sizes
python engines/image/scripts/download_models.py             # default model only, ~16GB
python engines/image/scripts/download_models.py --all       # every model + LoRAs, ~27GB
```

Files already present at the right size are skipped, so this is safe to re-run
and safe to interrupt. Weights are staged through `.cache/` at the repo root
rather than the system drive.

Requires `huggingface_hub` in whichever Python you run it with:
`pip install huggingface_hub Pillow numpy`.

**Note on sources.** `black-forest-labs/FLUX.1-schnell` is gated on Hugging Face
and returns 401 without an accepted licence and a token. The Flux UNET and VAE
therefore come from equivalent public mirrors and are renamed to the filenames
the workflows expect. Nothing here needs a Hugging Face token.

### Start the server

```bash
cd engines/image && ./python_embeded/python.exe -s ComfyUI/main.py --windows-standalone-build --listen 127.0.0.1 --port 8188
```

Confirm with `curl -s http://127.0.0.1:8188/system_stats`. Leave it running;
every image script waits for it and will time out with a clear message if it is
not up.

## 2. Audio engine

The audio engine needs **Python 3.11** and its own virtualenv. It cannot share
an environment with the image engine, because AudioCraft pins an older torch.

### Python 3.11

AudioCraft does not install on 3.13 or 3.14. On Windows with scoop:

```bash
scoop bucket add versions
scoop install python311
```

### Virtualenv and dependencies

Install order matters. AudioCraft must go in with `--no-deps`, because its own
pin of `av` has no wheel for Python 3.11 on Windows and fails to compile.

```bash
cd engines/audio
"$HOME/scoop/apps/python311/current/python.exe" -m venv venv
./venv/Scripts/python.exe -m pip install --upgrade pip

# Pinned trio first: the only combination with prebuilt Windows wheels.
./venv/Scripts/pip.exe install torch==2.4.0 torchaudio==2.4.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# AudioCraft without its dependency resolution, then what it needs at runtime.
./venv/Scripts/pip.exe install --no-deps audiocraft==1.3.0
./venv/Scripts/pip.exe install "numpy<2" av einops encodec flashy hydra-core hydra_colorlog \
    julius librosa num2words omegaconf protobuf sentencepiece soundfile spacy \
    torchmetrics transformers
```

Verify:

```bash
./venv/Scripts/python.exe -c "import torch, xformers; from audiocraft.models import MusicGen, AudioGen; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.4.0+cu121 True` and no import error.

Audio models download on first generation, roughly 6GB per medium model, into
`.cache/` at the repo root. There is no separate download step.

### Why these pins

Each one is a workaround for a specific failure, so do not float them:

| Pin | Without it |
|---|---|
| `torch 2.4.0` + `xformers 0.0.27.post2` | No prebuilt Windows wheel for any newer pairing. Building xformers from source fails on Windows path-length limits inside its vendored flash-attention tree. |
| `numpy < 2` | torch 2.4.0 was built against numpy 1.x. With numpy 2 present, torch raises "Numpy is not available" from inside AudioCraft's sampling loop. |
| `audiocraft --no-deps` | Its `av` pin has no Python 3.11 Windows wheel and fails to compile. |

`engines/audio/requirements.txt` records the resolved versions.

## 3. Verify the install

```bash
python engines/image/scripts/asset_presets.py --preset item_icon \
    --subject "a red health potion" --output out/image/potion.png

./engines/audio/venv/Scripts/python.exe engines/audio/scripts/audio_presets.py \
    --preset ui_click --output out/audio/click.wav
```

The first image render after a server restart takes about 13 seconds while the
model loads, then about 5. The first audio generation downloads its model.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ComfyUI did not answer on http://127.0.0.1:8188 within 600s` | The server is not running. Start it as above. |
| `ComfyUI rejected the workflow` with a `value not in list` error | The model file is missing. Run `download_models.py --all`. |
| `ModuleNotFoundError: No module named 'audiocraft'` | You ran an audio script with the wrong interpreter. Use `engines/audio/venv/Scripts/python.exe`. |
| `RuntimeError: Numpy is not available` | numpy 2 crept into the audio venv. `pip install "numpy<2"`. |
| `ModuleNotFoundError: No module named 'xformers'` | The pinned pairing was not installed. See above; newer versions have no wheel here. |
| CUDA out of memory during audio | Use `musicgen-small`, or make sure ComfyUI is not holding the GPU. Only one engine should generate at a time. |
| `No module named 'triton'` warning | Harmless on Windows. Ignore it. |
| A sprite comes out tiny in a large transparent frame | Background keying left something opaque, widening the crop. Check the raw frame saved next to the output. |

## Note on the two engines and one GPU

Nothing stops you running both at once, but a 16GB card cannot comfortably hold
Flux and an AudioCraft medium model simultaneously. If audio generation fails
with an out-of-memory error, stop the ComfyUI server first.
