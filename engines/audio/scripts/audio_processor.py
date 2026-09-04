#!/usr/bin/env python3
"""Post-processing that turns a raw generation into a usable game asset.

Raw MusicGen and AudioGen output is not shippable. It starts with silence, ends
mid-phrase, sits at whatever level the model felt like, and clicks when a game
starts or stops it. This module is the audio counterpart of the image engine's
pixel_art_processor: the model provides material, this makes it an asset.

  trim        strip leading and trailing near-silence. For a sound effect this
              is the difference between a responsive hit and a late one, since
              a leading 200ms of silence is 200ms of input lag the player feels
  normalise   bring everything to one level so a whole SFX folder can be played
              at a single mixer volume without per-file tweaking
  fade        a few ms in and out, because a waveform cut mid-cycle produces an
              audible click on playback
  loop        for music, cross-fade the tail back over the head so the track
              repeats without a seam

That loop step is the same trick the image engine uses to make a texture tile:
blend the end into the beginning so the wrap point is continuous. The measure
of success is the same too, a discontinuity at the seam no larger than the
signal's own frame-to-frame variation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

GAME_RATE = 44100  # what engines expect; both models output lower


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Adequate here because we only ever upsample."""
    if src_rate == dst_rate:
        return audio
    n = int(round(len(audio) * dst_rate / src_rate))
    src_t = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n, endpoint=False)
    if audio.ndim == 1:
        return np.interp(dst_t, src_t, audio).astype(np.float32)
    return np.stack([np.interp(dst_t, src_t, audio[:, c]) for c in range(audio.shape[1])],
                    axis=1).astype(np.float32)


def trim_silence(audio: np.ndarray, rate: int, threshold_db: float = -45.0,
                 pad_ms: float = 8.0) -> np.ndarray:
    """Drop near-silent head and tail, keeping a short pad so nothing clips off."""
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    threshold = 10 ** (threshold_db / 20.0)

    # Short-window RMS rather than raw samples, so a single stray sample does
    # not defeat the trim.
    win = max(1, int(rate * 0.005))
    trimmed = mono[: len(mono) - len(mono) % win].reshape(-1, win)
    loud = np.sqrt((trimmed ** 2).mean(axis=1)) > threshold
    if not loud.any():
        return audio

    first, last = int(np.argmax(loud)), int(len(loud) - np.argmax(loud[::-1]))
    pad = int(rate * pad_ms / 1000.0)
    start = max(0, first * win - pad)
    end = min(len(mono), last * win + pad)
    return audio[start:end]


def normalise(audio: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Scale so the loudest sample sits at `peak_db`, leaving a little headroom."""
    peak = float(np.abs(audio).max())
    if peak < 1e-9:
        return audio
    return (audio * (10 ** (peak_db / 20.0) / peak)).astype(np.float32)


def fade(audio: np.ndarray, rate: int, in_ms: float = 5.0, out_ms: float = 15.0) -> np.ndarray:
    """Taper both ends so playback cannot click."""
    out = audio.copy()
    n_in, n_out = int(rate * in_ms / 1000.0), int(rate * out_ms / 1000.0)
    if n_in and n_in < len(out):
        ramp = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        out[:n_in] = (out[:n_in].T * ramp).T
    if n_out and n_out < len(out):
        ramp = np.linspace(1.0, 0.0, n_out, dtype=np.float32)
        out[-n_out:] = (out[-n_out:].T * ramp).T
    return out


def make_loop(audio: np.ndarray, rate: int, crossfade_ms: float = 400.0) -> np.ndarray:
    """Cross-fade the tail back over the head so the track repeats seamlessly.

    The result is shorter than the input by the cross-fade length: those samples
    are consumed making the join. Equal-power curves are used rather than linear
    ones, so the blend does not dip in perceived loudness halfway through.
    """
    n = int(rate * crossfade_ms / 1000.0)
    if n * 2 >= len(audio):
        return audio

    head, tail = audio[:n], audio[-n:]
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_in, fade_out = np.sqrt(t), np.sqrt(1.0 - t)

    if audio.ndim == 1:
        joined = tail * fade_out + head * fade_in
    else:
        joined = (tail.T * fade_out).T + (head.T * fade_in).T

    body = audio[n:-n]
    return np.concatenate([joined, body]).astype(np.float32)


def loop_seam_ratio(audio: np.ndarray) -> float:
    """How discontinuous the wrap is, against the signal's own average step.

    1.0 means looping is as smooth as the waveform's normal motion. The image
    engine measures tile seams the same way.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if len(mono) < 3:
        return 0.0
    seam = abs(float(mono[0]) - float(mono[-1]))
    inner = float(np.abs(np.diff(mono)).mean())
    return seam / max(inner, 1e-9)


def process(audio: np.ndarray, rate: int, kind: str = "sfx",
            target_rate: int = GAME_RATE, loop: bool | None = None,
            crossfade_ms: float = 400.0, peak_db: float = -1.0) -> tuple[np.ndarray, dict]:
    """Full chain. `kind` picks defaults appropriate to music or effects."""
    loop = (kind == "music") if loop is None else loop
    info = {"in_rate": rate, "in_seconds": len(audio) / rate}

    audio = np.asarray(audio, dtype=np.float32)
    audio = trim_silence(audio, rate)
    audio = resample(audio, rate, target_rate)

    if loop:
        audio = make_loop(audio, target_rate, crossfade_ms)
        # A looping track must not fade out, or the seam becomes a dropout.
        audio = fade(audio, target_rate, in_ms=0.0, out_ms=0.0)
    else:
        audio = fade(audio, target_rate)

    audio = normalise(audio, peak_db)
    info.update(out_rate=target_rate, out_seconds=len(audio) / target_rate,
                looped=loop, seam_ratio=loop_seam_ratio(audio) if loop else None)
    return audio, info


def save(audio: np.ndarray, rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 16-bit PCM WAV: what every engine loads without a decoder, and the right
    # choice for short effects where decode latency matters.
    sf.write(str(path), audio, rate, subtype="PCM_16")


def main() -> int:
    ap = argparse.ArgumentParser(description="Post-process a generated audio file.")
    ap.add_argument("--input", "-i", type=Path, required=True)
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--kind", choices=("music", "sfx"), default="sfx")
    ap.add_argument("--loop", action="store_true", help="force loop processing")
    ap.add_argument("--no-loop", action="store_true", help="force no loop processing")
    ap.add_argument("--crossfade-ms", type=float, default=400.0)
    ap.add_argument("--rate", type=int, default=GAME_RATE)
    args = ap.parse_args()

    audio, rate = sf.read(str(args.input), dtype="float32")
    loop = True if args.loop else (False if args.no_loop else None)
    out, info = process(audio, rate, args.kind, args.rate, loop, args.crossfade_ms)
    save(out, args.rate, args.output)

    seam = f", seam {info['seam_ratio']:.2f}" if info["seam_ratio"] is not None else ""
    print(f"{args.input.name} {info['in_seconds']:.1f}s @{info['in_rate']}Hz -> "
          f"{info['out_seconds']:.1f}s @{info['out_rate']}Hz"
          f"{' looped' if info['looped'] else ''}{seam} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
