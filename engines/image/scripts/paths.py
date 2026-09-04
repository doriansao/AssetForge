#!/usr/bin/env python3
"""Filesystem layout for the image engine.

Every path is derived from this file's own location, so the repository can be
cloned anywhere without editing scripts. Nothing here should ever be an
absolute path typed by hand.
"""
from __future__ import annotations

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ENGINE = SCRIPTS.parent                  # engines/image
ROOT = ENGINE.parent.parent              # repository root

COMFY = ENGINE / "ComfyUI"
MODELS = COMFY / "models"
WORKFLOWS = ENGINE / "workflows"
PYTHON = ENGINE / "python_embeded" / "python.exe"

OUT = ROOT / "out" / "image"
RAW = OUT / "raw"                        # full-resolution renders
TILES = OUT / "tiles"                    # finished tileset
CACHE = ROOT / ".cache"                  # model download staging

SERVER = "http://127.0.0.1:8188"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
