#!/usr/bin/env python3
"""Filesystem layout for the audio engine.

Mirrors engines/image/scripts/paths.py: every path derives from this file's
location so the repository can be cloned anywhere.
"""
from __future__ import annotations

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ENGINE = SCRIPTS.parent                  # engines/audio
ROOT = ENGINE.parent.parent              # repository root

VENV_PYTHON = ENGINE / "venv" / "Scripts" / "python.exe"
OUT = ROOT / "out" / "audio"
CACHE = ROOT / ".cache"                  # shared with the image engine


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
