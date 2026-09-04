#!/usr/bin/env python3
"""Registry of audio models the engine can drive.

Two families, and they are not interchangeable. MusicGen was trained on music
and produces tonal, rhythmic material; AudioGen was trained on environmental
sound and produces impacts, whooshes and ambience. Asking MusicGen for a sword
hit gives you a short piece of music that vaguely resembles one.

Both come from Meta's AudioCraft and share one Python API, which is why they
are here. Sample rates differ (MusicGen 32kHz, AudioGen 16kHz) and the
post-processor resamples both to a game-friendly rate.

VRAM figures are for fp16 on this machine's 16GB card. Only one model is
resident at a time.
"""
from __future__ import annotations

MODELS = {
    # --- music ---------------------------------------------------------
    "musicgen-small": {
        "repo": "facebook/musicgen-small",
        "kind": "music",
        "sample_rate": 32000,
        "params": "300M",
        "vram": "~2 GB",
        "notes": "Fast drafts and loops. Weakest musicality; good for iterating on a brief.",
    },
    "musicgen-medium": {
        "repo": "facebook/musicgen-medium",
        "kind": "music",
        "sample_rate": 32000,
        "params": "1.5B",
        "vram": "~6 GB",
        "notes": "Default for music. Best quality-per-second on a 16GB card.",
    },
    "musicgen-large": {
        "repo": "facebook/musicgen-large",
        "kind": "music",
        "sample_rate": 32000,
        "params": "3.3B",
        "vram": "~12 GB",
        "notes": "Highest quality, noticeably slower. Use for final theme music only.",
    },
    "musicgen-melody": {
        "repo": "facebook/musicgen-melody",
        "kind": "music",
        "sample_rate": 32000,
        "params": "1.5B",
        "vram": "~6 GB",
        "notes": "Accepts a reference melody as well as text. Use to make "
                 "variations of one theme for different areas, which is the "
                 "cheapest way to make a soundtrack feel composed rather than "
                 "assembled.",
    },
    # --- sound effects -------------------------------------------------
    "audiogen-medium": {
        "repo": "facebook/audiogen-medium",
        "kind": "sfx",
        "sample_rate": 16000,
        "params": "1.5B",
        "vram": "~6 GB",
        "notes": "Default for sound effects. 16kHz output is the main limit: "
                 "fine for impacts, thuds and ambience, thin for anything that "
                 "needs bright high end like glass or cymbals.",
    },
}

DEFAULTS = {"music": "musicgen-medium", "sfx": "audiogen-medium"}

# Models worth trying that are NOT wired up here, with the reason. Kept in the
# registry so the decision is recorded next to the code rather than lost.
ALTERNATIVES = {
    "stabilityai/stable-audio-open-1.0": (
        "Best open model for sound effects: 44.1kHz stereo, up to 47s, and much "
        "brighter than AudioGen. Gated on Hugging Face, so it needs an accepted "
        "licence and a token, and it runs through stable-audio-tools rather than "
        "AudioCraft. Worth adding if you hit AudioGen's 16kHz ceiling."),
    "cvssp/audioldm2": (
        "Ungated text-to-audio via diffusers, 16kHz. Similar quality band to "
        "AudioGen with an easier install; useful as a second opinion on a prompt."),
    "suno/bark": (
        "Speech and vocal-adjacent sounds, including non-verbal noises. The "
        "practical choice for creature vocalisations and barks, which neither "
        "MusicGen nor AudioGen handles well."),
}


def get(name: str) -> dict:
    if name not in MODELS:
        raise SystemExit(f"unknown audio model {name!r}; available: {', '.join(MODELS)}")
    return MODELS[name]
