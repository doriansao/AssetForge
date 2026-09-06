#!/usr/bin/env python3
"""Image to textured GLB with TRELLIS.2, through the mesh engine's own ComfyUI.

The mesh engine is a second ComfyUI checkout with its own Python 3.11 venv
(torch 2.7 + cu128), because the TRELLIS.2 node ships compiled wheels for that
pairing and the image engine runs Python 3.13 / torch 2.13. Never install one
engine's packages into the other. Start it with:

    engines/mesh/venv/Scripts/python.exe engines/mesh/ComfyUI/main.py --listen 127.0.0.1 --port 8189

Then:

    generate_mesh.py --image saucer_rgba.png --output out/mesh/saucer.glb
        [--faces 200000] [--texture 2048] [--pipeline 1024_cascade] [--seed 7]
        [--back back.png --left left.png --right right.png]   # multi-view

Node inputs are filled from the server's /object_info defaults and then
overridden, so this stays correct when the node author adds options.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request

SERVER = os.environ.get("ASSETFORGE_MESH_SERVER", "http://127.0.0.1:8189")
HERE = os.path.dirname(os.path.abspath(__file__))


def api(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{SERVER}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def wait_for_server(timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            api("/system_stats")
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"mesh ComfyUI did not answer on {SERVER} within {timeout}s")


def upload_image(path):
    import io
    name = f"mesh_{random.randint(0, 1 << 30)}_{os.path.basename(path)}"
    boundary = "----assetforge"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
               f"Content-Type: image/png\r\n\r\n".encode())
    body.write(open(path, "rb").read())
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{SERVER}/upload/image", data=body.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["name"]


_INFO = None


def node(class_type, **overrides):
    """Build an API node with every widget at the server's default, then override."""
    global _INFO
    if _INFO is None:
        _INFO = api("/object_info")
    spec = _INFO[class_type]["input"]
    inputs = {}
    for section in ("required", "optional"):
        for name, desc in spec.get(section, {}).items():
            kind = desc[0]
            opts = desc[1] if len(desc) > 1 and isinstance(desc[1], dict) else {}
            if isinstance(kind, list):            # combo
                inputs[name] = opts.get("default", kind[0] if kind else None)
            elif kind in ("INT", "FLOAT", "STRING", "BOOLEAN") and "default" in opts:
                inputs[name] = opts["default"]
            elif kind == "BOOLEAN":
                inputs[name] = False
            elif kind in ("INT", "FLOAT"):
                inputs[name] = 0
            elif kind == "STRING":
                inputs[name] = ""
            # links (IMAGE, MESH...) come from overrides
    inputs.update(overrides)
    # drop optional link inputs that were never wired
    for name in list(inputs):
        if inputs[name] is None:
            del inputs[name]
    return {"class_type": class_type, "inputs": inputs}


def build(front, seed, faces, texture, pipeline, back=None, left=None, right=None, prefix="assetforge_mesh"):
    wf = {
        "1": node("Trellis2LoadModel", modelname="microsoft/TRELLIS.2-4B", backend="sdpa",
                  sparse_backend="xformers", conv_backend="flex_gemm", low_vram=True),
        "2": node("Trellis2LoadImageWithTransparency", image=front),
        "3": node("Trellis2PreProcessImage", image=["2", 2], padding=10, remove_background=False, max_size=1024),
    }
    if back or left or right:
        extra = {}
        for key, img in (("back_image", back), ("left_image", left), ("right_image", right)):
            if img:
                wf[f"2{key[0]}"] = node("Trellis2LoadImageWithTransparency", image=img)
                wf[f"3{key[0]}"] = node("Trellis2PreProcessImage", image=[f"2{key[0]}", 2], padding=10,
                                        remove_background=False, max_size=1024)
                extra[key] = [f"3{key[0]}", 0]
        wf["4"] = node("Trellis2MeshWithVoxelMultiViewGenerator", pipeline=["1", 0], front_image=["3", 0],
                       seed=seed, pipeline_type=pipeline, generate_texture_slat=True, **extra)
    else:
        wf["4"] = node("Trellis2MeshWithVoxelGenerator", pipeline=["1", 0], image=["3", 0], seed=seed,
                       pipeline_type=pipeline, generate_texture_slat=True)
    wf["5"] = node("Trellis2FillHolesWithCuMesh", mesh=["4", 0])
    wf["6"] = node("Trellis2SimplifyMesh", mesh=["5", 0], target_face_num=faces, method="Cumesh")
    wf["7"] = node("Trellis2UnWrapAndRasterizer", mesh=["6", 0], bvh=["4", 1], texture_size=texture)
    wf["8"] = node("Trellis2ExportMesh", trimesh=["7", 0], filename_prefix=prefix, file_format="glb")
    return wf


def poll(prompt_id, timeout=3600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = api(f"/history/{prompt_id}").get(prompt_id)
        if entry:
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                raise SystemExit("mesh generation failed:\n" + json.dumps(st.get("messages", []), indent=1)[:4000])
            # Trellis2ExportMesh is not an OUTPUT_NODE, so history carries no
            # "outputs" for it; completion status is the only signal.
            if st.get("completed") or st.get("status_str") == "success":
                return entry
        time.sleep(3)
    raise SystemExit("timed out waiting for the mesh")


def main():
    p = argparse.ArgumentParser(description="TRELLIS.2 image to GLB through the mesh ComfyUI.")
    p.add_argument("--image", required=True, help="front view, RGBA with transparent background")
    p.add_argument("--back"), p.add_argument("--left"), p.add_argument("--right")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--faces", type=int, default=200000)
    p.add_argument("--texture", type=int, default=2048)
    p.add_argument("--pipeline", default="1024_cascade", choices=("512", "1024", "1024_cascade", "1536_cascade"))
    a = p.parse_args()
    seed = a.seed if a.seed is not None else random.randint(0, 2**31 - 1)
    wait_for_server()
    names = {k: (upload_image(v) if v else None) for k, v in
             (("front", a.image), ("back", a.back), ("left", a.left), ("right", a.right))}
    prefix = f"assetforge_mesh_{random.randint(0, 1 << 30):x}"
    wf = build(names["front"], seed, a.faces, a.texture, a.pipeline, names["back"], names["left"], names["right"],
               prefix=prefix)
    t = time.time()
    try:
        pid = api("/prompt", {"prompt": wf, "client_id": "mesh"})["prompt_id"]
    except urllib.error.HTTPError as exc:
        raise SystemExit("mesh ComfyUI rejected the workflow:\n" + exc.read().decode(errors="replace")[:4000])
    print(f"queued {pid} (seed {seed}, {a.pipeline})", flush=True)
    poll(pid)
    # The export node writes <prefix>_NNNNN_.glb into the mesh ComfyUI output folder.
    import glob
    out_dir = os.path.join(HERE, "..", "ComfyUI", "output")
    hits = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.glb")), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"finished but no {prefix}_*.glb in {os.path.abspath(out_dir)}")
    glb = hits[-1]
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    shutil.copyfile(glb, a.output)
    print(f"wrote {a.output} ({os.path.getsize(a.output)/2**20:.1f} MiB) in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
