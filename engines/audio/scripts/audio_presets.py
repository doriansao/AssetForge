#!/usr/bin/env python3
"""Recipes for the audio a 2D game actually needs.

The prompts here are longer and more physical than you would expect. That is
deliberate: both models respond to material and space far more than to a name.
"sword hit" produces something generic, while "a steel blade striking a wooden
shield, sharp crack with a short woody decay, close and dry" produces a usable
effect. Naming the material, the impact, the decay and the room is the whole
technique.

Durations are short on purpose. A UI click wants a fraction of a second, and
generating four seconds then trimming wastes time and invites the model to add
material you do not want.

Loops are on for music and off for effects. A looping effect would cross-fade
its own tail over its attack, which destroys the transient that makes it read
as an impact.

Usage:
  python audio_presets.py --list
  <venv python> audio_presets.py --preset sword_hit --output out/audio/sword.wav
  <venv python> audio_presets.py --preset town_theme --output out/audio/town.wav
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# kind, duration seconds, prompt
PRESETS = {
    # --- interface -----------------------------------------------------
    "ui_click": ("sfx", 1.0,
        "a short soft mechanical click, single clean transient, dry and close, no reverb"),
    "ui_confirm": ("sfx", 1.5,
        "a bright short positive chime, two ascending bell tones, clean and dry"),
    "ui_cancel": ("sfx", 1.5,
        "a short low descending blip, muted and dry, negative feedback tone"),
    "ui_error": ("sfx", 1.5,
        "a harsh short buzz, low dissonant electronic error tone, dry"),
    # --- pickups and rewards -------------------------------------------
    "pickup_coin": ("sfx", 1.0,
        "a bright metallic coin ping, single small bell-like transient, quick decay"),
    "pickup_item": ("sfx", 1.5,
        "a soft magical shimmer, ascending sparkle, light and bright, short"),
    "powerup": ("sfx", 2.5,
        "a rising magical whoosh building to a bright chime, energetic and clean"),
    # --- movement ------------------------------------------------------
    "jump": ("sfx", 1.0,
        "a short airy whoosh with a light spring, quick upward motion, dry"),
    "land": ("sfx", 1.0,
        "a soft dull thud of boots landing on packed earth, short and dry"),
    "footstep_stone": ("sfx", 1.0,
        "a single boot step on a stone floor, hard heel click with a short slap, dry and close"),
    "footstep_grass": ("sfx", 1.0,
        "a single soft footstep on grass and dry leaves, light rustle, close and dry"),
    # --- combat --------------------------------------------------------
    "sword_swing": ("sfx", 1.5,
        "a fast blade swishing through air, sharp whoosh, close and dry, no impact"),
    "sword_hit": ("sfx", 1.5,
        "a steel blade striking a wooden shield, sharp crack with a short woody decay, close and dry"),
    "punch_impact": ("sfx", 1.0,
        "a heavy fist impact on a body, dull low thud with a short slap, close and dry"),
    "enemy_death": ("sfx", 2.0,
        "a low guttural creature groan falling in pitch then collapsing to silence"),
    "player_hurt": ("sfx", 1.5,
        "a short sharp pained grunt, human, close and dry"),
    # --- world ---------------------------------------------------------
    "explosion": ("sfx", 3.0,
        "a large explosion, deep low boom with a sharp crack and a long rumbling tail, debris"),
    "fire_loop": ("sfx", 4.0,
        "a steady crackling campfire, continuous burning and popping embers, close"),
    "water_splash": ("sfx", 2.0,
        "a body falling into water, heavy splash with bubbling and dripping afterwards"),
    "door_open": ("sfx", 2.5,
        "a heavy wooden door creaking slowly open on iron hinges in a stone corridor"),
    "chest_open": ("sfx", 2.0,
        "a wooden chest lid creaking open with a metal latch clunk, close and dry"),
    # --- music ---------------------------------------------------------
    "town_theme": ("music", 20.0,
        "a warm cheerful medieval village theme, acoustic lute, soft flute and light "
        "percussion, gentle major key, relaxed mid tempo, looping game music"),
    "battle_theme": ("music", 20.0,
        "an urgent orchestral battle theme, driving percussion, brass stabs and fast "
        "strings, minor key, high energy, looping game music"),
    "boss_theme": ("music", 20.0,
        "a heavy dramatic boss battle theme, low choir, pounding timpani and dissonant "
        "brass, dark and menacing, looping game music"),
    "menu_theme": ("music", 20.0,
        "a calm atmospheric main menu theme, soft synth pads, sparse piano, slow and "
        "spacious, gentle and inviting, looping game music"),
    "dungeon_ambient": ("music", 20.0,
        "a dark ambient dungeon drone, low sustained strings, distant echoing drips, "
        "tense and sparse, minimal melody, looping game music"),
    "victory_fanfare": ("music", 8.0,
        "a short triumphant orchestral victory fanfare, bright brass and cymbal, "
        "major key, celebratory ending"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preset")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--count", type=int, default=1, help="generate N variations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--extra", default="", help="appended to the preset prompt")
    args = ap.parse_args()

    if args.list:
        for name, (kind, dur, prompt) in PRESETS.items():
            print(f"{name:18s} {kind:5s} {dur:4.1f}s  {prompt[:64]}...")
        return 0

    if not (args.preset and args.output):
        raise SystemExit("--preset and --output are required")
    if args.preset not in PRESETS:
        raise SystemExit(f"unknown preset {args.preset!r}; try --list")

    import generate_audio as gen
    import audio_processor as post
    import models as registry

    kind, duration, prompt = PRESETS[args.preset]
    if args.extra:
        prompt = f"{prompt}, {args.extra}"
    model_name = args.model or registry.DEFAULTS[kind]

    clips, rate = gen.generate(prompt, model_name, duration, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for i, clip in enumerate(clips):
        target = args.output if args.count == 1 else \
            args.output.with_name(f"{args.output.stem}_{i+1:02d}{args.output.suffix}")
        processed, info = post.process(clip, rate, kind)
        post.save(processed, post.GAME_RATE, target)
        seam = f", seam {info['seam_ratio']:.2f}" if info["seam_ratio"] is not None else ""
        print(f"{args.preset}: {target.name}  {info['out_seconds']:.1f}s"
              f"{' looped' if info['looped'] else ''}{seam}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
