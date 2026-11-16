"""Step 1c: remove seams and repeat until the image has the target size.

Each iteration:
    1. compute the energy map of the CURRENT image   (seamcarve.energy)
    2. find the minimum seam with DP                  (seamcarve.dp)
    3. delete that seam's pixels -> image is one pixel narrower (or shorter)

The energy is recomputed every time because removing a seam changes the
neighbours of the pixels next to it, so their energy changes too.

Order of seams when BOTH width and height shrink: we alternate a vertical
seam and a horizontal seam while both dimensions still need reducing, then
finish whichever dimension remains.

Horizontal seams are handled by transposing: removing a horizontal seam from
an image is the same as removing a vertical seam from its transpose
(rows <-> columns), then transposing back.

Complexity for an m x n image reduced by k_w columns and k_h rows:
    each removal costs O(m·n) (energy + DP + copy), so the total is
    O((k_w + k_h) · m · n). Extra space is O(m·n) for the energy map and DP
    tables of the current image.
    (Only pixels next to the removed seam actually change energy, so a
    faster version could update the energy locally; we keep the simple
    full recomputation, as the assignment describes.)
"""

from typing import Callable

import numpy as np

from seamcarve.dp import find_vertical_seam
from seamcarve.energy import dual_gradient_energy

# Type of a seam finder: takes an (m, n) energy map, returns m column indices.
# Any algorithm with this signature can be plugged into resize().
SeamFinder = Callable[[np.ndarray], np.ndarray]
EnergyFunction = Callable[[np.ndarray], np.ndarray]


def remove_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Delete one pixel per row: the pixel at column seam[i] in row i.

    Args:
        img:  (m, n) or (m, n, c) array.
        seam: m column indices, one per row.

    Returns:
        A new array of shape (m, n-1) or (m, n-1, c). The input is not modified.
    """
    m, n = img.shape[:2]
    seam = np.asarray(seam)
    if n < 2:
        raise ValueError("cannot remove a seam from an image that is 1 pixel wide")
    if seam.shape != (m,):
        raise ValueError(f"seam must have one entry per row ({m}), got shape {seam.shape}")
    if np.any((seam < 0) | (seam >= n)):
        raise ValueError("seam has a column index outside the image")

    # Boolean mask: True = keep, False = remove. Exactly one False per row.
    keep = np.ones((m, n), dtype=bool)
    keep[np.arange(m), seam] = False

    # img[keep] picks the kept pixels row by row (row-major order), giving a
    # flat list of m*(n-1) pixels; reshape restores the rows. The channel
    # axis (if any) is carried along unchanged.
    return img[keep].reshape((m, n - 1) + img.shape[2:])


def remove_horizontal_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Delete one pixel per column: the pixel at row seam[j] in column j.

    Implemented by transposing (swap rows and columns), removing a vertical
    seam, and transposing back.
    """
    transposed = np.swapaxes(img, 0, 1)
    return np.swapaxes(remove_vertical_seam(transposed, seam), 0, 1)


def resize(
    img: np.ndarray,
    new_height: int,
    new_width: int,
    finder: SeamFinder = find_vertical_seam,
    energy_fn: EnergyFunction = dual_gradient_energy,
) -> np.ndarray:
    """Shrink an image to (new_height, new_width) by seam carving.

    Args:
        img:        (m, n) or (m, n, c) image, e.g. uint8 RGB.
        new_height: target number of rows,    1 <= new_height <= m.
        new_width:  target number of columns, 1 <= new_width  <= n.
        finder:     returns a minimum vertical seam for an energy map
                    (default: the dynamic-programming version).
        energy_fn:  computes the energy map of an image
                    (default: dual-gradient energy).

    Returns:
        New array of shape (new_height, new_width[, c]) with the same dtype.
    """
    height, width = img.shape[:2]
    if not (1 <= new_height <= height and 1 <= new_width <= width):
        raise ValueError(
            f"target size must be between 1x1 and the current size {height}x{width}, "
            f"got {new_height}x{new_width} (this implementation only shrinks images)"
        )

    out = img
    next_is_vertical = True
    while out.shape[1] > new_width or out.shape[0] > new_height:
        need_vertical = out.shape[1] > new_width
        need_horizontal = out.shape[0] > new_height

        # Alternate while both are needed; otherwise do whichever is left.
        if need_vertical and (next_is_vertical or not need_horizontal):
            out = _remove_one_vertical(out, finder, energy_fn)
        else:
            out = _remove_one_horizontal(out, finder, energy_fn)
        next_is_vertical = not next_is_vertical

    return np.ascontiguousarray(out)


def _remove_one_vertical(img, finder, energy_fn):
    energy = energy_fn(img)       # 1. energy of the current image
    seam = finder(energy)         # 2. minimum vertical seam
    return remove_vertical_seam(img, seam)   # 3. delete it


def _remove_one_horizontal(img, finder, energy_fn):
    # Work on the transpose so the same vertical finder can be reused:
    # a horizontal seam of img is a vertical seam of img transposed.
    transposed = np.swapaxes(img, 0, 1)
    reduced = _remove_one_vertical(transposed, finder, energy_fn)
    return np.swapaxes(reduced, 0, 1)
