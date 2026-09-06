# Mesh engine: image to 3D to 2.5D sprites

The third engine. It turns one keyed render into a textured GLB with
Microsoft's TRELLIS.2, then renders that mesh from any camera with Blender.
This is the "Dead Cells" pipeline (3D model, 2D sprites) with the modelling
step replaced by a diffusion model, and it is the only route here to sprites
that are *guaranteed* consistent across angles and frames: a rotating turret,
a hatch that opens, a hull hit-flash seen from any side.

Measured on the Cowduction mothership: one 1024-cascade mesh with a 2048px
texture took 350 s on the RTX 4060 Ti (16GB), and 24 orthographic sprite
frames at 768px took Blender about 4 minutes with Cycles on the GPU.

## Why a separate engine

TRELLIS.2 needs compiled CUDA extensions (cumesh, nvdiffrast, flex_gemm,
o_voxel, nvdiffrec_render). The ComfyUI node that wraps it ships Windows
wheels only for Python 3.11 with torch 2.7.0 + cu128 (and 3.12 / torch 2.8,
2.10). The image engine runs Python 3.13 with torch 2.13. Mixing them breaks
one or the other, so the mesh engine is a second ComfyUI checkout with its own
venv on port 8189. Never install its packages into the image engine's
`python_embeded`, and never the reverse.

```
engines/mesh/
  venv/               Python 3.11 + torch 2.7.0+cu128 (not in git)
  ComfyUI/            second checkout, port 8189 (not in git)
  ComfyUI-Trellis2/   visualbruno's node, cloned with its wheels (not in git)
  setup_mesh_env.sh   installs everything above into the venv
  scripts/
    generate_mesh.py  image -> GLB through the mesh ComfyUI
    render_sprites.py Blender: GLB -> sprite frames + sheet
```

## Setup

1. Python 3.11 on the path (scoop's `python311` works), then:

   ```bash
   cd engines/mesh
   python3.11 -m venv venv
   ./venv/Scripts/python.exe -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
   git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
   git clone --depth 1 https://github.com/visualbruno/ComfyUI-Trellis2.git
   cd ../.. && bash engines/mesh/setup_mesh_env.sh
   ```

   Pin torchaudio explicitly: ComfyUI's requirements otherwise pull the
   newest torchaudio, whose extension fails to load against torch 2.7 with
   `WinError 127`, and ComfyUI then refuses to start.

2. Weights. The node downloads on first use, but that happens inside a
   ComfyUI job and holds the GPU queue, so fetch them up front:

   ```bash
   ./engines/mesh/venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download as s; s('microsoft/TRELLIS.2-4B', local_dir='engines/mesh/ComfyUI/models/microsoft/TRELLIS.2-4B')"
   ```

   plus `model.safetensors`, `config.json` and `preprocessor_config.json` from
   the node's DINOv3 mirror into
   `engines/mesh/ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m/`.
   About 16 GB and 1.2 GB. TRELLIS.2 is MIT. DINOv3 is Meta's own licence,
   gated on Hugging Face; the node defaults to a public mirror. Read that
   licence before shipping anything derived from it.

3. Blender 4.2 or newer on the machine (5.2 LTS tested). The renderer is a
   Blender-python script, so no add-ons.

4. Start the server: `engines/mesh/venv/Scripts/python.exe engines/mesh/ComfyUI/main.py --listen 127.0.0.1 --port 8189`.
   The node takes about 40 s to import. Confirm with
   `curl -s http://127.0.0.1:8189/object_info | grep -c Trellis2` (78 nodes).

## Use

```bash
# 1. key the render (BiRefNet, see IMAGE.md)
python engines/image/scripts/remove_background.py --input render.png --output hull_rgba.png --crop
# 2. mesh it
./engines/mesh/venv/Scripts/python.exe engines/mesh/scripts/generate_mesh.py --image hull_rgba.png --output out/mesh/hull.glb --faces 150000
# 3. sprite it
"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b -P engines/mesh/scripts/render_sprites.py -- \
    --glb out/mesh/hull.glb --out out/mesh/sprites/hull --size 768 --azimuths 8 --elevations 0,20,60 --sheet
```

`generate_mesh.py` fills every node input from the server's `/object_info`
defaults and overrides only what matters (`sdpa` attention and `xformers`
sparse attention, because flash-attn has no Windows wheel for this torch;
`1024_cascade` pipeline; `low_vram` on). `--back/--left/--right` switch to the
multi-view generator, which is where Qwen's Multiple Angles LoRA (IMAGE.md)
pays off: derive the side and back views from the hero render first and the
mesh stops guessing what the far side looks like.

`render_sprites.py` orbits an orthographic camera round the mesh's bounding
sphere so every frame shares one pivot, writes RGBA PNGs on a transparent
film, and with `--sheet` packs a horizontal strip plus a JSON of frame boxes
and camera angles. Orthographic is the default on purpose: a 2.5D game
composites a fixed vertical projection, and perspective would make the top of
a tall hull lean differently from its base.

## Traps

- **The export node is not an output node.** ComfyUI's history never lists
  outputs for it, so a poller waiting on `outputs` hangs forever after the
  job has finished. `generate_mesh.py` waits on the completion status and then
  finds the GLB by its unique filename prefix in `engines/mesh/ComfyUI/output/`.
- **Feed it RGBA.** `Trellis2LoadImageWithTransparency` uses the alpha as the
  object mask. A render on a white plate produces a slab. Key it first.
- **12 million faces before simplification.** The 1024 cascade produces very
  dense meshes; `--faces` (default 200k) is what keeps the GLB usable.
- **Don't run it alongside a 12GB image-engine job.** The shape and texture
  stages each want most of the card. Serialise mesh jobs and Qwen-Image-Edit.
