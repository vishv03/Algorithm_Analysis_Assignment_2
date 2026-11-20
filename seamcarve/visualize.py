"""Bonus: seam tracking for step-by-step visualisation and interactive resizing.

Every seam is found in the CURRENT (already carved) image, so its column
indices are not positions in the original picture. To draw all the removed
seams on the original image we carry an "index map" along with the image:

    index[i, j] = flat position (row * n + col) in the ORIGINAL image of the
                  pixel that is now at (i, j).

Whenever a seam is removed from the image, the same seam is removed from the
index map, so index always tells us where every remaining pixel came from.
Looking up the seam in the index map before removing it gives the seam's
pixels in original coordinates. Cost: one extra O(m·n) integer copy per seam,
the same order as removing the seam from the image itself.

Removal-order map (Avidan & Shamir 2007, "multi-size images")
    order[i, j] = the step (0-based) at which original pixel (i, j) was removed,
                  or k (the total number of steps) if it was never removed.

Removing vertical seams down to width w removes exactly the pixels with
order < n - w, one per row per step. So once the map is computed (one carving
pass, O(k·m·n)), ANY width between the minimum and n can be produced instantly
with a mask: keep the pixels with order >= n - w. This is what makes the
interactive resizer (interactive.py) respond in real time. The result is
exactly resize(img, m, w): the first n - w steps of carving to the minimum
width are the same steps as carving straight to w.
"""

from typing import Iterator

import numpy as np

from seamcarve.carve import (
    EnergyFunction,
    SeamFinder,
    SeamStep,
    carve_steps,
    remove_horizontal_seam,
    remove_vertical_seam,
)
from seamcarve.dp import find_vertical_seam
from seamcarve.energy import dual_gradient_energy

# Colours of the seam overlays: early seams use the first colour, late seams
# the second, with a linear blend in between.
VERTICAL_COLORS = ((255, 0, 0), (255, 220, 0))      # red -> yellow
HORIZONTAL_COLORS = ((0, 70, 255), (0, 230, 255))   # blue -> cyan


def track_seams(
    img: np.ndarray,
    new_height: int,
    new_width: int,
    finder: SeamFinder = find_vertical_seam,
    energy_fn: EnergyFunction = dual_gradient_energy,
    incremental: bool = False,
) -> Iterator[tuple[SeamStep, np.ndarray, np.ndarray]]:
    """carve_steps() plus the position of every seam in the ORIGINAL image.

    Yields (step, rows, cols): rows[k], cols[k] are the original coordinates
    of the k-th pixel of the seam removed in that step.
    """
    m, n = img.shape[:2]
    index = np.arange(m * n).reshape(m, n)
    for step in carve_steps(img, new_height, new_width, finder, energy_fn, incremental):
        if step.direction == "vertical":
            flat = index[np.arange(index.shape[0]), step.seam]
            index = remove_vertical_seam(index, step.seam)
        else:
            flat = index[step.seam, np.arange(index.shape[1])]
            index = remove_horizontal_seam(index, step.seam)
        rows, cols = np.divmod(flat, n)
        yield step, rows, cols


def removal_order(
    img: np.ndarray,
    min_size: int,
    direction: str = "vertical",
    finder: SeamFinder = find_vertical_seam,
    energy_fn: EnergyFunction = dual_gradient_energy,
    incremental: bool = False,
) -> np.ndarray:
    """Removal-order map for seams in ONE direction, down to `min_size`.

    Args:
        img:       (m, n[, c]) image.
        min_size:  smallest width (vertical) or height (horizontal) needed.
        direction: "vertical" (shrink the width) or "horizontal" (the height).

    Returns:
        int32 (m, n): order[i, j] = step at which pixel (i, j) is removed,
        or the total number of steps if it survives down to min_size.
    """
    m, n = img.shape[:2]
    if direction == "vertical":
        steps = track_seams(img, m, min_size, finder, energy_fn, incremental)
        total = n - min_size
    elif direction == "horizontal":
        steps = track_seams(img, min_size, n, finder, energy_fn, incremental)
        total = m - min_size
    else:
        raise ValueError(f"direction must be 'vertical' or 'horizontal', got {direction!r}")

    order = np.full((m, n), total, dtype=np.int32)
    for step, rows, cols in steps:
        order[rows, cols] = step.number - 1
    return order


def apply_removal_order(img: np.ndarray, order: np.ndarray, size: int,
                        direction: str = "vertical") -> np.ndarray:
    """The image carved to `size` columns (vertical) or rows (horizontal).

    O(m·n): a single mask, no seam search. `size` must be at least the
    min_size the order map was computed for.
    """
    m, n = img.shape[:2]
    if direction == "horizontal":
        img_t = np.swapaxes(img, 0, 1)
        return np.swapaxes(apply_removal_order(img_t, order.T, size, "vertical"), 0, 1)

    total = int(order.max())       # pixels that are never removed carry `total`
    if not (n - total <= size <= n):
        raise ValueError(f"size must be between {n - total} and {n}, got {size}")
    keep = order >= n - size       # exactly `size` pixels per row
    return img[keep].reshape((m, size) + img.shape[2:])


def seam_color(position: float, colors) -> np.ndarray:
    """Blend between colors[0] (position 0) and colors[1] (position 1)."""
    start, end = (np.asarray(c, dtype=np.float64) for c in colors)
    return np.round(start + (end - start) * position).astype(np.uint8)


def energy_to_image(energy: np.ndarray) -> np.ndarray:
    """Energy map as a viewable grayscale RGB uint8 image.

    The square root and the 99th-percentile clip stop a few very strong edges
    from making everything else black.
    """
    e = np.sqrt(np.maximum(energy, 0))
    top = np.percentile(e, 99) or 1.0
    gray = np.clip(e / top, 0, 1) * 255
    return np.repeat(gray.astype(np.uint8)[:, :, np.newaxis], 3, axis=2)


def draw_seam(canvas: np.ndarray, seam: np.ndarray, direction: str, color) -> None:
    """Paint one seam (in canvas coordinates) onto an RGB canvas, in place."""
    if direction == "vertical":
        canvas[np.arange(seam.size), seam] = color
    else:
        canvas[seam, np.arange(seam.size)] = color


def all_seams_overlay(img: np.ndarray, removed) -> np.ndarray:
    """The original image with every removed seam painted on it.

    Args:
        img:     original (m, n, 3) uint8 image.
        removed: list of (direction, rows, cols) in original coordinates,
                 in the order the seams were removed.
    """
    canvas = np.array(img, dtype=np.uint8, copy=True)
    counts = {"vertical": 0, "horizontal": 0}
    for direction, _, _ in removed:
        counts[direction] += 1
    seen = {"vertical": 0, "horizontal": 0}
    for direction, rows, cols in removed:
        colors = VERTICAL_COLORS if direction == "vertical" else HORIZONTAL_COLORS
        position = seen[direction] / max(counts[direction] - 1, 1)
        canvas[rows, cols] = seam_color(position, colors)
        seen[direction] += 1
    return canvas
