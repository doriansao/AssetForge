#!/usr/bin/env python3
"""Pixel-art post-processing for the Flux asset pipeline.

The grid-detection and quantisation core is vendored from
doriansao/codex-pixel-art-generator (MIT, Dorian Butron), which is itself a
faithful Python port of Hugo Duprez's spritefusion-pixel-snapper
(MIT, https://github.com/Hugo-Dz/spritefusion-pixel-snapper).

The important idea, and the reason this beats a plain downscale: Flux does not
draw pixel art, it draws a picture OF pixel art. The blocks it paints are
roughly uniform but drift off any fixed grid, and their edges are anti-aliased.
Resizing to 48x48 with any filter therefore samples across block boundaries and
turns a one-pixel outline into a smear. Instead this module measures where the
blocks actually are, from image gradients, and takes one colour per cell:

  1. k-means++ quantise at FULL resolution, so outline pixels get their own
     centroid before any spatial decision is made
  2. build 1D gradient profiles per axis
  3. estimate the native block size from the median peak-to-peak distance
  4. walk each profile in float, snapping cuts to gradient peaks
  5. take the per-cell MODE of RGBA, so a one-pixel outline wins its cell
     instead of being averaged away

Added here on top of the vendored core, because the upstream skill gets these
from other tools in its own workflow:

  * background keying to alpha (upstream uses a chroma-key helper belonging to
    the codex imagegen skill, which is not part of that repository)
  * content cropping, so a sprite fills its target grid
  * a target-size bridge: upstream DETECTS the output resolution and has no
    concept of "make this 48x48", so a requested size is converted into the
    pixel-size override that the grid walker accepts
  * integer NEAREST upscaling for preview

Standalone use:
  python pixel_art_processor.py -i raw.png -o sprite.png --target-size 48
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Vendored core -- see the module docstring for provenance. Do not edit
# casually: the float step size and the alpha>=1 k-means training set are both
# load bearing for outline crispness, and both were regressions fixed upstream.
# ---------------------------------------------------------------------------

def compute_gradient_profile(arr_rgba: np.ndarray, axis: int) -> np.ndarray:
    """1D gradient profile along the given axis. Hugo's compute_profiles.

    axis=1 → vertical cuts (gradient along X, summed over Y)
    axis=0 → horizontal cuts (gradient along Y, summed over X)

    Kernel is [-1, 0, 1] on grayscale. Per Hugo: pixels with alpha=0 contribute
    gray=0; pixels with alpha>0 contribute FULL gray (no alpha attenuation).
    This matters along soft-matte silhouette edges — alpha-attenuated gray
    would weaken the silhouette-edge peak and let it drift away from logical
    pixel boundaries.
    """
    rgb = arr_rgba[..., :3].astype(np.float64)
    alpha = arr_rgba[..., 3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray[alpha == 0] = 0.0

    if axis == 1:
        grad = np.zeros_like(gray)
        grad[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
        profile = grad.sum(axis=0)
    else:
        grad = np.zeros_like(gray)
        grad[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
        profile = grad.sum(axis=1)
    return profile


def estimate_step_size(
    profile: np.ndarray,
    peak_threshold_multiplier: float = 0.2,
    peak_distance_filter: int = 4,
) -> float | None:
    """Estimate the native pixel step size from a gradient profile.

    Faithful port of Hugo's estimate_step_size:
      - threshold = max(profile) * peak_threshold_multiplier (default 0.2)
      - find local maxima above threshold
      - filter peaks closer than peak_distance_filter - 1 = 3 apart
      - return median of consecutive-peak differences as FLOAT

    Returning a float (not int) matters: Hugo carries f64 step sizes through
    `walk` so the walker stays locked on the true logical-pixel grid across
    long sweeps. Integer-truncating drifts the walker off the grid over a
    few-hundred-pixel character, smearing thin features (especially 1-pixel
    outlines) across cell boundaries.
    """
    n = len(profile)
    if n < 3 or profile.max() <= 0:
        return None
    threshold = float(profile.max()) * peak_threshold_multiplier

    # Local maxima above threshold
    raw_peaks: list[int] = []
    for i in range(1, n - 1):
        v = profile[i]
        if v > threshold and v > profile[i - 1] and v > profile[i + 1]:
            raw_peaks.append(i)

    # Filter peaks closer than peak_distance_filter - 1 apart.
    # Hugo's filter keeps the FIRST of each cluster (not the tallest) — match
    # exactly so output is deterministic and matches the upstream reference.
    filtered: list[int] = []
    for p in raw_peaks:
        if not filtered or p - filtered[-1] > (peak_distance_filter - 1):
            filtered.append(p)

    if len(filtered) < 2:
        return None

    diffs = np.diff(np.array(filtered, dtype=np.float64))
    return float(np.median(diffs))


def walk(
    profile: np.ndarray,
    step_size: float,
    window_ratio: float = 0.35,
    min_search_window: float = 2.0,
    strength_threshold: float = 0.5,
) -> list[int]:
    """Walk along the profile, snapping each expected cut to its local peak.

    Faithful port of Hugo's walk:
      - Maintain `current_pos` as float, advance by `target = current_pos + step`.
      - Search window is `max(step * window_ratio, min_search_window)`.
      - Within the window, find the argmax. If that peak's value is above
        `mean(profile) * strength_threshold`, snap to it; otherwise place a
        uniform cut at `target` (don't be deflected by sub-mean noise peaks
        in smooth regions).

    The float walk + strength threshold combo is what keeps the cuts aligned
    with logical pixel boundaries across the full sprite without getting
    pulled off-grid by smooth-region noise. Cells stay aligned with actual
    color transitions, so 1-pixel outlines fill exactly one cell instead of
    smearing across two.
    """
    n = len(profile)
    if step_size <= 1 or n <= step_size:
        return [0, n]

    search_window = max(step_size * window_ratio, min_search_window)
    mean_val = float(profile.sum() / max(1, len(profile)))
    threshold = mean_val * strength_threshold

    cuts: list[int] = [0]
    current_pos = 0.0

    while current_pos < n:
        target = current_pos + step_size
        if target >= n:
            cuts.append(n)
            break

        lo = max(int(target - search_window), int(current_pos + 1.0))
        hi = min(int(target + search_window), n)
        if hi <= lo:
            current_pos = target
            continue

        window = profile[lo:hi]
        rel = int(np.argmax(window))
        peak_idx = lo + rel
        peak_val = float(window[rel])

        if peak_val > threshold:
            cuts.append(peak_idx)
            current_pos = float(peak_idx)
        else:
            uniform_cut = int(target)
            if uniform_cut <= cuts[-1]:
                uniform_cut = cuts[-1] + 1
            cuts.append(min(uniform_cut, n))
            current_pos = target

    if cuts[-1] != n:
        cuts.append(n)
    return cuts


def stabilize_axis(
    cuts: list[int], step_size: float, n: int, min_gap_ratio: float = 0.5
) -> list[int]:
    """Stabilize cuts: drop cuts that are too close to their neighbor.

    Simplified version of Hugo's cross-axis stabilize. Removes cuts that
    would produce cells smaller than half the step size — those are
    detection artifacts.
    """
    if len(cuts) < 3:
        return cuts
    min_gap = max(1, int(round(step_size * min_gap_ratio)))
    stabilized = [cuts[0]]
    for c in cuts[1:-1]:
        if c - stabilized[-1] >= min_gap:
            stabilized.append(c)
    if cuts[-1] - stabilized[-1] >= min_gap:
        stabilized.append(cuts[-1])
    else:
        stabilized[-1] = cuts[-1]
    return stabilized


def snap_uniform_cuts(
    profile: np.ndarray,
    limit: int,
    target_step: float,
    window_ratio: float = 0.35,
    min_search_window: float = 2.0,
    strength_threshold: float = 0.5,
    min_required: int = 4,
) -> list[int]:
    """Fallback when the walker yields too few cuts.

    Computes uniformly-spaced target positions and snaps each toward its
    local peak (same strength gate as `walk`). Used when a profile is too
    weak / noisy for the walker to find enough cuts naturally. Faithful
    port of Hugo's snap_uniform_cuts.
    """
    if limit <= 1:
        return [0, max(limit, 1)]

    if target_step is None or not np.isfinite(target_step) or target_step <= 0:
        desired_cells = 1
    else:
        desired_cells = max(1, int(round(limit / target_step)))
    desired_cells = max(desired_cells, max(min_required - 1, 1))
    desired_cells = min(desired_cells, limit)

    cell_width = limit / desired_cells
    search_window = max(cell_width * window_ratio, min_search_window)
    mean_val = float(profile.sum() / max(1, len(profile))) if len(profile) else 0.0
    threshold = mean_val * strength_threshold

    cuts: list[int] = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break
        lo = max(int(target - search_window), prev + 1)
        hi = min(int(target + search_window) + 1, limit)
        if hi <= lo:
            lo = prev + 1
            hi = lo + 1
        window = profile[lo:hi] if hi <= len(profile) else profile[lo:]
        if window.size == 0:
            chosen = min(int(round(target)), limit - 1)
        else:
            rel = int(np.argmax(window))
            peak_idx = lo + rel
            peak_val = float(window[rel])
            if peak_val >= threshold:
                chosen = peak_idx
            else:
                chosen = int(round(target))
        if chosen <= prev:
            chosen = prev + 1
        if chosen >= limit:
            chosen = limit - 1
        cuts.append(chosen)

    if cuts[-1] != limit:
        cuts.append(limit)

    # Deduplicate while preserving order
    seen: set[int] = set()
    dedup: list[int] = []
    for c in cuts:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def detect_grid(
    arr_rgba: np.ndarray,
    pixel_size_override: float | None = None,
    min_cuts_per_axis: int = 4,
    fallback_target_segments: int = 64,
    max_step_ratio: float = 1.8,
) -> tuple[list[int], list[int], float]:
    """Detect grid cuts (Hugo's full algorithm, faithful port).

    Returns (col_cuts, row_cuts, effective_step_size). col_cuts defines the
    vertical cell boundaries (X positions); row_cuts the horizontal ones.

    Step sizes are tracked as floats throughout to avoid drift over long
    sweeps (a 0.5-px-per-step error compounds to several pixels by the time
    the walker reaches the far edge of the sprite).
    """
    h, w = arr_rgba.shape[:2]

    prof_x = compute_gradient_profile(arr_rgba, axis=1)
    prof_y = compute_gradient_profile(arr_rgba, axis=0)

    if pixel_size_override is not None:
        step_x = step_y = float(pixel_size_override)
    else:
        sx = estimate_step_size(prof_x)
        sy = estimate_step_size(prof_y)
        # Hugo's resolve_step_sizes: average unless ratio > max_step_ratio
        if sx is None and sy is None:
            fallback = max(1.0, min(h, w) / fallback_target_segments)
            step_x = step_y = fallback
        elif sx is None:
            step_x = step_y = float(sy)
        elif sy is None:
            step_x = step_y = float(sx)
        else:
            ratio = max(sx, sy) / max(1e-9, min(sx, sy))
            if ratio > max_step_ratio:
                step_x = step_y = float(min(sx, sy))
            else:
                step_x = step_y = float((sx + sy) / 2.0)

    col_cuts = walk(prof_x, step_x)
    row_cuts = walk(prof_y, step_y)

    col_cuts = stabilize_axis(col_cuts, step_x, w)
    row_cuts = stabilize_axis(row_cuts, step_y, h)

    # Hugo's fallback: if a profile yielded too few cuts (weak gradients in
    # smooth sprites), snap to uniform cuts at the resolved step size.
    if len(col_cuts) < min_cuts_per_axis:
        col_cuts = snap_uniform_cuts(prof_x, w, step_x, min_required=min_cuts_per_axis)
    if len(row_cuts) < min_cuts_per_axis:
        row_cuts = snap_uniform_cuts(prof_y, h, step_y, min_required=min_cuts_per_axis)

    return col_cuts, row_cuts, step_x


def kmeans_plus_plus(
    pixels: np.ndarray,
    k: int,
    max_iter: int = 15,
    tol: float = 0.01,
    seed: int = 42,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """K-means++ initialization + Lloyd's iterations on RGB pixels.

    Matches Hugo's algorithm:
      - K-means++ init with weighted-by-squared-distance sampling.
      - Lloyd iterations: max 15, early exit when max centroid movement
        (squared) drops below 0.01.
      - Squared-Euclidean RGB distance (no perceptual weighting).
      - Deterministic seed (42 by default).

    Args:
        pixels: (N, 3) float32 array of RGB values in 0-255 range.
        k: number of clusters.
        max_iter: max Lloyd iterations.
        tol: convergence threshold on max squared centroid movement.
        seed: RNG seed for reproducibility.
        chunk_size: distance-computation chunk size to bound memory.

    Returns:
        (centroids (k, 3) float32, labels (N,) int32)
    """
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    n = pixels.shape[0]
    rng = np.random.default_rng(seed)

    if n == 0:
        return np.zeros((k, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    if n <= k:
        centroids = np.zeros((k, 3), dtype=np.float32)
        centroids[:n] = pixels
        # Fill unused centroids with the last pixel (won't get assignments)
        if n < k and n > 0:
            centroids[n:] = pixels[-1]
        return centroids, np.arange(n, dtype=np.int32)

    # ---- k-means++ init ----
    centroids = np.empty((k, 3), dtype=np.float32)
    idx0 = int(rng.integers(n))
    centroids[0] = pixels[idx0]

    # Running min squared-distance to any existing centroid
    min_d_sq = np.full(n, np.inf, dtype=np.float32)

    for ci in range(1, k):
        # Update min_d_sq with the just-added centroid (only)
        latest = centroids[ci - 1]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            d = pixels[start:end] - latest
            d_sq = (d * d).sum(axis=-1)
            np.minimum(min_d_sq[start:end], d_sq, out=min_d_sq[start:end])

        total = float(min_d_sq.sum())
        if total <= 0.0:
            idx = int(rng.integers(n))
        else:
            r = float(rng.random()) * total
            cum = np.cumsum(min_d_sq)
            idx = int(np.searchsorted(cum, r))
            if idx >= n:
                idx = n - 1
        centroids[ci] = pixels[idx]

    # ---- Lloyd iterations ----
    labels = np.empty(n, dtype=np.int32)

    def assign_labels(c: np.ndarray) -> None:
        # |x - c|^2 = |x|^2 + |c|^2 - 2 x·c. Use chunked dot products.
        c_sq = (c * c).sum(axis=-1)  # (k,)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = pixels[start:end]
            # (chunk, k) = -2 * chunk @ c.T + c_sq (broadcast)
            # We can skip x.x because it's constant per row → argmin is the same.
            dots = chunk @ c.T  # (chunk, k)
            dist_proxy = c_sq[None, :] - 2.0 * dots  # (chunk, k), shifted by |x|^2
            labels[start:end] = np.argmin(dist_proxy, axis=1)

    for _ in range(max_iter):
        assign_labels(centroids)

        new_centroids = centroids.copy()
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                new_centroids[ci] = pixels[mask].mean(axis=0)
            # else: keep old centroid (empty cluster)

        diff = new_centroids - centroids
        max_move_sq = float((diff * diff).sum(axis=-1).max())
        centroids = new_centroids
        if max_move_sq < tol:
            break

    # Final assignment (in case loop ended without re-labeling at last centroids)
    assign_labels(centroids)
    return centroids, labels


# ---------------------------------------------------------------------------
# Pipeline layer added for this project
# ---------------------------------------------------------------------------


# Chroma keys the generator can ask for, as the two channels that must dominate
# the third. Named rather than given as RGB because the model never returns the
# exact colour asked for -- one request for #FF00FF came back as (184,55,114) --
# and hue dominance is what matters, not the value.
CHROMA_KEYS = {
    "magenta": (0, 2, 1),   # R and B over G
    "green": (1, 1, 0),     # G over R and B  (second index unused)
}


def detect_chroma(rgb: np.ndarray, margin: int = 40, coverage: float = 0.5) -> str | None:
    """Say which chroma key the border is painted in, if any."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.int32)
    r, g, b = border[:, 0], border[:, 1], border[:, 2]
    if ((r - g > margin) & (b - g > margin)).mean() > coverage:
        return "magenta"
    if ((g - r > margin) & (g - b > margin)).mean() > coverage:
        return "green"
    return None


def key_chroma_to_alpha(img: Image.Image, kind: str, margin: int = 40,
                        contract: int = 1) -> Image.Image:
    """Key out a chroma background by hue dominance, at any brightness.

    This is why chroma keys exist, and it solves a problem no amount of
    post-processing on a white background does. A cast shadow falling on the key
    is a *darker version of the key colour*, so a test on hue rather than on
    value removes the subject's shadow along with the background, in one step.
    On a white field the same shadow is a pale grey that is not the background
    colour, survives keying, and then has to be hunted down by heuristics that
    cannot reliably tell a shadow from a pair of boots.

    Choose a key whose strong channels are not the subject's. Magenta is the
    default and suits the cool palettes here; use green for warm or red-dominant
    subjects, which would otherwise lose pixels to a magenta test.

    `contract` erodes the matte by a pixel to drop the key-tinted fringe around
    the silhouette, which is cheaper and more predictable than despill.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.int32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    if kind == "magenta":
        bg = (r - g > margin) & (b - g > margin)
    elif kind == "green":
        bg = (g - r > margin) & (g - b > margin)
    else:
        raise ValueError(f"unknown chroma key {kind!r}")

    keep = ~bg
    if contract > 0:
        size = contract * 2 + 1
        mask = Image.fromarray((keep * 255).astype(np.uint8), "L")
        keep = np.asarray(mask.filter(ImageFilter.MinFilter(size))) > 0

    out = np.dstack([rgb.astype(np.uint8), np.where(keep, 255, 0).astype(np.uint8)])
    out[~keep] = (0, 0, 0, 0)
    return Image.fromarray(out, "RGBA")


def key_background_to_alpha(img: Image.Image, tolerance: int = 130,
                            enclosed: bool = True) -> Image.Image:
    """Flood the flat background in from the border and turn it into alpha.

    The upstream skill asks the model for a chroma-key background and strips it
    with a dedicated helper. Flux here is prompted for a plain solid background
    of no fixed colour, so the key colour is sampled from the border instead of
    being assumed.

    Flood fill rather than a global colour threshold, so background-coloured
    pixels inside the subject survive.

    The default tolerance is deliberately loose. A contact shadow on a white
    field is a soft grey that shades continuously out of the background, so a
    fill wide enough to walk into it removes the shadow as part of the
    background, while the subject stays far outside the threshold because it is
    darker and saturated. Measured on one render: at tolerance 72 the sprite was
    904px tall and ended in a 2px shadow tail; at 130 it was 809px and ended
    flat on the boots, with the opaque pixel count essentially unchanged. The
    window from about 120 to 240 is stable; past 300 it starts eating the
    subject.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    arr = np.asarray(rgb)

    # Modal colour around the 1px border is the key.
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    colours, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
    key = colours[counts.argmax()].astype(int)

    # Pick a sentinel the image cannot be confused with.
    sentinel = (255, 0, 255) if abs(key - np.array([255, 0, 255])).sum() > 90 else (0, 255, 0)
    work = rgb.copy()
    anchors = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
               (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]

    for ax, ay in anchors:
        # Search a small window around each anchor for a pixel that actually
        # matches the key, rather than trusting the single pixel there. A lone
        # artefact pixel is common in the corners -- one SDXL render had a
        # (166,255,255) speck in the bottom-left, which caused that whole seed
        # to be skipped and left two opaque columns at the image edge. Those
        # columns then widened the content crop from x448 to x0 and shrank the
        # sprite to a fraction of its frame.
        seed = None
        for dx in range(0, 16, 2):
            for dy in range(0, 16, 2):
                px, py = min(max(ax + (dx if ax == 0 else -dx), 0), w - 1), \
                         min(max(ay + (dy if ay == 0 else -dy), 0), h - 1)
                if abs(np.array(rgb.getpixel((px, py)), dtype=int) - key).sum() <= tolerance:
                    seed = (px, py)
                    break
            if seed:
                break
        if seed and not np.all(np.array(work.getpixel(seed)) == np.array(sentinel)):
            ImageDraw.floodfill(work, seed, sentinel, thresh=tolerance)

    filled = np.asarray(work)
    bg = np.all(filled == np.array(sentinel), axis=-1)
    if enclosed:
        bg = clear_enclosed_background(arr, bg, key, tolerance)

    out = np.dstack([arr, np.where(bg, 0, 255).astype(np.uint8)])
    out[bg] = (0, 0, 0, 0)
    return Image.fromarray(out, "RGBA")


def _components(mask: np.ndarray):
    """Yield 4-connected components of a boolean mask as coordinate lists."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack, comp = [(int(sy), int(sx))], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        yield comp


def clear_enclosed_background(rgb: np.ndarray, bg: np.ndarray, key: np.ndarray,
                              tolerance: int, min_frac: float = 0.0015) -> np.ndarray:
    """Also key out background-coloured regions the border flood fill cannot reach.

    A flood fill entering from the edge stops at the silhouette, so any hole
    fully enclosed by the subject stays opaque: the gap between a creature's
    front legs, the space inside a fence, the loop of a handle. Those regions
    then composite into a scene as solid blocks of background colour.

    Only regions above `min_frac` of the frame are cleared. Small specks of
    near-white are usually genuine highlights -- an eye glint, a metal
    specular -- and removing those would punch holes in the art.
    """
    near = (np.abs(rgb.astype(np.int32) - key).sum(-1) <= tolerance) & ~bg
    if not near.any():
        return bg
    min_area = max(24, int(min_frac * near.size))
    for comp in _components(near):
        if len(comp) >= min_area:
            ys, xs = np.array(comp).T
            bg[ys, xs] = True
    return bg


def trim_ground_plinth(img: Image.Image, lighten: float = 1.25,
                       max_frac: float = 0.22) -> Image.Image:
    """Remove the ground shadow or plinth drawn beneath an isolated subject.

    Diffusion models put a contact shadow, a dirt disc or a stone base under an
    object even when told not to, and Flux Schnell ignores negative prompts
    entirely so it cannot be suppressed at generation time. It survives
    background keying because it is not the background colour, and then does two
    kinds of damage: it dirties the sprite, and because it sits below the feet it
    becomes the bottom of the bounding box, so bottom-anchored placement rests
    the *shadow* on the ground and the character floats above it.

    Detection is by brightness, not by shape. A shadow cast onto a white field is
    a pale grey or tan, while the subject is darker and more saturated, and this
    separates cleanly in practice: on one render the body averaged luminance 64
    against 126 for the shadow rows, and on another 100 against 182. Rows are
    removed from the bottom up while they stay far lighter than the body, and the
    scan stops at the first row that does not qualify.

    Shape-based detection was tried first and abandoned. Width alone cannot tell
    a shadow from a pair of boots: the narrowest row in the search band is
    usually the bottom edge of the shadow itself, and searching further up finds
    the gap between a character's legs and cuts the legs off. Brightness does not
    have that failure mode, because legs are as dark as the body.
    """
    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[..., 3] > 8
    if not alpha.any():
        return img

    lum = rgba[..., :3].astype(np.float32).mean(-1)
    h = rgba.shape[0]
    limit = int(h * (1.0 - max_frac))

    # Reference brightness from the subject proper, excluding the band that
    # might be plinth, so a large plinth cannot drag the reference up.
    body = alpha.copy()
    body[limit:] = False
    if not body.any():
        return img
    reference = float(np.median(lum[body])) * lighten

    # Collect every row in the band that reads as plinth, then cut at the
    # highest one. Stopping at the first row that fails going upwards is not
    # enough: a shadow is densest, and therefore darkest, in its middle, so the
    # scan halts partway and leaves the top of the shadow attached.
    bright = [k for k in range(limit, h)
              if alpha[k].any() and float(lum[k][alpha[k]].mean()) > reference]
    if not bright:
        return img

    cut = min(bright)
    span = h - cut
    # Guard against a single stray bright row high in the band dragging the cut
    # up: most of what is being removed has to look like plinth.
    if span <= 0 or len(bright) / span < 0.6:
        return img
    return img.crop((0, 0, img.width, cut))


def crop_to_content(img: Image.Image, pad: int = 0, despeckle: int = 5) -> Image.Image:
    """Trim transparent margins so the subject fills its grid.

    The bounding box is taken from a despeckled copy of the alpha channel, so a
    few stray pixels left behind by imperfect keying cannot drag the crop out to
    the image edge. The returned pixels are the original ones; despeckling only
    decides where to cut.
    """
    rgba = img.convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]

    mask = Image.fromarray((alpha > 8).astype(np.uint8) * 255, "L")
    if despeckle > 1:
        # Opening: erode then dilate. Anything thinner than the filter vanishes.
        mask = mask.filter(ImageFilter.MinFilter(despeckle)).filter(
            ImageFilter.MaxFilter(despeckle))
    core = np.asarray(mask) > 0
    if not core.any():
        core = alpha > 8          # everything was speckle-thin; fall back
    if not core.any():
        return img

    ys, xs = np.nonzero(core)
    x0, x1 = max(0, int(xs.min()) - pad), min(img.width, int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(img.height, int(ys.max()) + 1 + pad)
    return rgba.crop((x0, y0, x1, y1))


def snap_to_grid(img: Image.Image, colors: int = 16, pixel_size: float | None = None,
                 alpha_threshold: int = 1, kmeans_seed: int = 42) -> tuple[Image.Image, float]:
    """Quantise, detect the native block grid, then take the mode of each cell.

    In-memory equivalent of the upstream snap_pixels entry point.
    """
    arr = np.array(img.convert("RGBA"))
    alpha = arr[..., 3]
    fg = alpha >= alpha_threshold

    rgb_q = arr[..., :3].copy()
    if fg.any():
        centroids, labels = kmeans_plus_plus(
            arr[fg, :3].astype(np.float32), k=colors, seed=kmeans_seed)
        rgb_q[fg] = np.clip(np.round(centroids), 0, 255).astype(np.uint8)[labels]
    rgb_q[~fg] = 0
    quantized = np.dstack([rgb_q, alpha])

    col_cuts, row_cuts, used_step = detect_grid(quantized, pixel_size)
    sh, sw = len(row_cuts) - 1, len(col_cuts) - 1

    packed = (quantized[..., 0].astype(np.uint32)
              | (quantized[..., 1].astype(np.uint32) << 8)
              | (quantized[..., 2].astype(np.uint32) << 16)
              | (quantized[..., 3].astype(np.uint32) << 24))

    out = np.zeros((sh, sw, 4), dtype=np.uint8)
    for y in range(sh):
        y0, y1 = row_cuts[y], row_cuts[y + 1]
        if y1 <= y0:
            continue
        for x in range(sw):
            x0, x1 = col_cuts[x], col_cuts[x + 1]
            if x1 <= x0:
                continue
            vals, counts = np.unique(packed[y0:y1, x0:x1].ravel(), return_counts=True)
            mode = int(vals[counts.argmax()])
            if mode == 0:
                continue  # transparent wins the cell
            out[y, x] = (mode & 0xFF, (mode >> 8) & 0xFF,
                         (mode >> 16) & 0xFF, (mode >> 24) & 0xFF)
    return Image.fromarray(out, "RGBA"), used_step


def nearest_upscale(img: Image.Image, scale: int | None = None,
                    canvas: tuple[int, int] | None = None) -> Image.Image:
    """Integer NEAREST upscale, by an explicit factor or fitted to a canvas.

    Never a fractional factor: that duplicates some source pixels more than
    others and the grid stops being regular.
    """
    if scale is not None:
        return img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    cw, ch = canvas or (1024, 1024)
    factor = max(1, min(cw // img.width, ch // img.height))
    big = img.resize((img.width * factor, img.height * factor), Image.NEAREST)
    out = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    out.paste(big, ((cw - big.width) // 2, (ch - big.height) // 2))
    return out


def process(img: Image.Image, target_size: int | None = 64, palette_size: int = 16,
            remove_background: bool = True, crop: bool = True,
            grid: str = "detect", alpha_threshold: int = 1,
            trim_plinth: bool = True) -> dict:
    """Full post-process. Returns the sprite plus what the detector decided.

    Upstream has no notion of a target size -- it reports whatever native
    resolution it detects -- so `target_size` (longest side) needs bridging.
    Two ways to do it, and the choice matters more than it looks:

    "detect" (default)
        Let the walker find the block size Flux actually drew, then NEAREST
        resample that clean sprite down to the target. The grid stays aligned
        with the render, so detail survives the trip.

    "force"
        Drive the walker at target_size cells directly, by handing it a
        pixel-size override. Only right when the render really is drawn on
        that grid; otherwise every cut lands slightly off the block boundaries
        and thin features (window frames, chimney edges) get eaten. Measured
        on a cottage render at 48px: forcing lost the window entirely, while
        detect-then-resample kept it.
    """
    if grid not in ("detect", "force"):
        raise ValueError(f"grid must be 'detect' or 'force', got {grid!r}")

    info: dict = {"input_size": img.size, "grid_mode": grid}

    if remove_background:
        # Prefer a chroma key when the render has one: it takes the shadow with
        # it. Fall back to the flood fill for plain white backgrounds, where the
        # shadow then has to be trimmed separately and less reliably.
        kind = detect_chroma(np.asarray(img.convert("RGB")))
        info["key"] = kind or "flood"
        if kind:
            img = key_chroma_to_alpha(img, kind)
        else:
            img = key_background_to_alpha(img)
            # Crop before trimming. The trim measures what fraction of the rows
            # it is about to remove actually look like plinth, and a full frame
            # has a band of empty rows under the subject that dilutes that
            # fraction below the guard, so the trim silently never fires.
            img = crop_to_content(img)
            if trim_plinth:
                before = img.height
                img = trim_ground_plinth(img)
                info["plinth_rows_removed"] = before - img.height
    else:
        img = img.convert("RGBA")
    if crop:
        img = crop_to_content(img)
    info["after_crop"] = img.size

    pixel_size = None
    if grid == "force" and target_size:
        pixel_size = max(1.0, max(img.size) / float(target_size))

    sprite, used_step = snap_to_grid(img, colors=palette_size, pixel_size=pixel_size,
                                     alpha_threshold=alpha_threshold)
    info["detected_step"] = used_step
    info["native_size"] = sprite.size

    if target_size and max(sprite.size) != target_size:
        ratio = target_size / float(max(sprite.size))
        sprite = sprite.resize((max(1, round(sprite.width * ratio)),
                                max(1, round(sprite.height * ratio))), Image.NEAREST)

    info["final_size"] = sprite.size
    opaque = np.asarray(sprite)[np.asarray(sprite)[..., 3] > 0][:, :3]
    info["colors"] = int(len(np.unique(opaque.reshape(-1, 3), axis=0))) if len(opaque) else 0
    info["sprite"] = sprite
    return info


def main() -> int:
    p = argparse.ArgumentParser(description="Snap an image onto a true pixel-art grid.")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument("--output", "-o", type=Path, required=True)
    p.add_argument("--target-size", type=int, default=64,
                   help="longest side of the sprite; 0 keeps the detected native size")
    p.add_argument("--palette-size", type=int, default=16, help="k-means colour count")
    p.add_argument("--upscale-factor", type=int, default=0,
                   help="also write a NEAREST preview scaled by this factor")
    p.add_argument("--keep-background", action="store_true")
    p.add_argument("--no-crop", action="store_true")
    p.add_argument("--keep-plinth", action="store_true",
                   help="keep the ground shadow the model drew under the subject")
    p.add_argument("--grid", choices=("detect", "force"), default="detect",
                   help="detect the native block size then resample (default), or "
                        "drive the walker at target-size cells directly")
    args = p.parse_args()

    info = process(Image.open(args.input), target_size=args.target_size or None,
                   palette_size=args.palette_size,
                   remove_background=not args.keep_background,
                   crop=not args.no_crop, grid=args.grid,
                   trim_plinth=not args.keep_plinth)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    info["sprite"].save(args.output)
    print(f"{args.input.name} {info['input_size'][0]}x{info['input_size'][1]}"
          f" -> native {info['native_size'][0]}x{info['native_size'][1]}"
          f" -> {info['final_size'][0]}x{info['final_size'][1]}"
          f"  ({info['colors']} colours, block {info['detected_step']:.2f}px)")

    if args.upscale_factor:
        prev = nearest_upscale(info["sprite"], scale=args.upscale_factor)
        prev_path = args.output.with_name(f"{args.output.stem}_x{args.upscale_factor}.png")
        prev.save(prev_path)
        print(f"preview {prev.width}x{prev.height} -> {prev_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
