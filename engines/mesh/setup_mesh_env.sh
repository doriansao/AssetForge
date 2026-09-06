#!/usr/bin/env bash
# Install the TRELLIS.2 stack into the mesh engine's own venv.
# Assumes venv exists with torch 2.7.0+cu128 (see docs/MESH.md), run from repo root:
#   bash engines/mesh/setup_mesh_env.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe
$PY -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== ComfyUI requirements"
$PY -m pip install -q -r ComfyUI/requirements.txt

echo "== TRELLIS.2 compiled wheels (Torch 2.7.0, cp311, win_amd64)"
W=ComfyUI-Trellis2/wheels/Windows/Torch270
for whl in cumesh-1.0 nvdiffrast-0.4.0 nvdiffrec_render-0.0.0 flex_gemm-0.0.1 o_voxel-0.0.1 custom_rasterizer-0.1; do
  $PY -m pip install -q "$W/${whl}-cp311-cp311-win_amd64.whl"
done

echo "== TRELLIS.2 python requirements + attention backend"
$PY -m pip install -q -r ComfyUI-Trellis2/requirements.txt
# The node's sparse attention accepts xformers or flash_attn; xformers has a
# Windows wheel for torch 2.7 cu128, flash_attn does not.
$PY -m pip install -q xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install -q huggingface_hub trimesh

echo "== link the node into the mesh ComfyUI"
mkdir -p ComfyUI/custom_nodes
if [ ! -e ComfyUI/custom_nodes/ComfyUI-Trellis2 ]; then
  cmd //c mklink //J "ComfyUI\\custom_nodes\\ComfyUI-Trellis2" "ComfyUI-Trellis2" >/dev/null
fi

echo "== import check"
$PY -c "import cumesh, nvdiffrast, flex_gemm, o_voxel, xformers; print('compiled deps import OK')"
echo "done. start with: engines/mesh/venv/Scripts/python.exe engines/mesh/ComfyUI/main.py --listen 127.0.0.1 --port 8189"
