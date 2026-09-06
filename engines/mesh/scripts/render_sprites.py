"""Render a GLB into 2.5D sprites with Blender, headless.

    blender -b -P render_sprites.py -- --glb ship.glb --out out/sprites/ship
        [--size 1024] [--azimuths 8] [--elevation 20] [--ortho] [--sheet]
        [--turntable 16]

Every frame is a PNG with alpha on a transparent film. The camera orbits the
object's bounding-sphere centre so every frame shares one pivot, which is
what a rotating turret or a mirrored faction hull needs. --sheet also packs
the frames into one horizontal strip with a JSON sidecar of frame boxes.

Why orthographic by default: 2.5D games composite sprites in a side view with
a fixed vertical projection (Cowduction's Turret.cs uses 0.62). Perspective
would make the top of a tall hull lean differently from its base.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--out", required=True, help="output prefix, frames go to <out>_<az>_<el>.png")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--azimuths", type=int, default=8, help="frames around the vertical axis")
    p.add_argument("--elevation", type=float, default=20.0, help="camera elevation in degrees")
    p.add_argument("--elevations", type=str, default=None,
                   help="comma list of elevations overriding --elevation, e.g. 0,20,60,90")
    p.add_argument("--start-azimuth", type=float, default=0.0)
    p.add_argument("--ortho", action="store_true", default=True)
    p.add_argument("--perspective", action="store_true")
    p.add_argument("--margin", type=float, default=1.08, help="ortho scale relative to bounding sphere")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--sheet", action="store_true")
    p.add_argument("--engine", default="CYCLES", choices=("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"))
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh objects in GLB")
    return meshes


def bounding_sphere(objs):
    pts = []
    for o in objs:
        for c in o.bound_box:
            pts.append(o.matrix_world @ Vector(c))
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    centre = (lo + hi) / 2
    radius = max((p - centre).length for p in pts)
    return centre, radius


def setup_render(a):
    sc = bpy.context.scene
    sc.render.engine = a.engine
    if a.engine == "CYCLES":
        sc.cycles.samples = a.samples
        sc.cycles.use_denoising = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            sc.cycles.device = "GPU"
        except Exception:
            pass
    sc.render.resolution_x = sc.render.resolution_y = a.size
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"


def setup_lights():
    # Soft studio key from front-top-left plus a fill, matching the concept renders.
    for name, rot, energy in [("key", (0.9, 0.0, -0.6), 3.0), ("fill", (1.1, 0.0, 2.2), 1.2)]:
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        ld.angle = math.radians(12)
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = rot
        bpy.context.scene.collection.objects.link(lo)
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.85, 0.88, 0.95, 1.0)
    bg.inputs[1].default_value = 0.6
    bpy.context.scene.world = world


def make_camera(a, centre, radius):
    cd = bpy.data.cameras.new("cam")
    if a.perspective:
        cd.type = "PERSP"
        cd.lens = 50
    else:
        cd.type = "ORTHO"
        cd.ortho_scale = 2 * radius * a.margin
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def place_camera(cam, centre, radius, az_deg, el_deg, perspective):
    az, el = math.radians(az_deg), math.radians(el_deg)
    dist = radius * (4.0 if perspective else 3.0)
    pos = centre + Vector((dist * math.cos(el) * math.sin(az),
                           -dist * math.cos(el) * math.cos(az),
                           dist * math.sin(el)))
    cam.location = pos
    direction = centre - pos
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main():
    a = parse_args()
    clear_scene()
    meshes = import_glb(a.glb)
    centre, radius = bounding_sphere(meshes)
    setup_render(a)
    setup_lights()
    cam = make_camera(a, centre, radius)
    elevations = [float(x) for x in a.elevations.split(",")] if a.elevations else [a.elevation]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    frames = []
    for el in elevations:
        for i in range(a.azimuths):
            az = a.start_azimuth + 360.0 * i / a.azimuths
            place_camera(cam, centre, radius, az, el, a.perspective)
            path = f"{a.out}_az{int(round(az)):03d}_el{int(round(el)):02d}.png"
            bpy.context.scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            frames.append({"file": os.path.basename(path), "azimuth": az, "elevation": el})
            print("rendered", path, flush=True)
    meta = {"glb": a.glb, "size": a.size, "ortho": not a.perspective, "centre": list(centre),
            "radius": radius, "frames": frames}
    with open(f"{a.out}_frames.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    if a.sheet:
        pack_sheet(a, frames)


def pack_sheet(a, frames):
    # Blender ships without PIL; use its own image API to build the strip.
    n = len(frames)
    sheet = bpy.data.images.new("sheet", a.size * n, a.size, alpha=True)
    buf = [0.0] * (a.size * n * a.size * 4)
    for k, f in enumerate(frames):
        img = bpy.data.images.load(os.path.join(os.path.dirname(os.path.abspath(a.out)), f["file"]))
        px = list(img.pixels)
        for y in range(a.size):
            src = y * a.size * 4
            dst = (y * a.size * n + k * a.size) * 4
            buf[dst:dst + a.size * 4] = px[src:src + a.size * 4]
        f["box"] = [k * a.size, 0, a.size, a.size]
        bpy.data.images.remove(img)
    sheet.pixels = buf
    sheet.filepath_raw = f"{a.out}_sheet.png"
    sheet.file_format = "PNG"
    sheet.save()
    with open(f"{a.out}_frames.json", "w") as fh:
        json.dump({"sheet": os.path.basename(sheet.filepath_raw), "frames": frames}, fh, indent=1)
    print("sheet", sheet.filepath_raw)


if __name__ == "__main__":
    main()
