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
    Improvement (resize(..., incremental=True)): only pixels next to the
    removed seam actually change energy, so we can remove the seam from the
    energy map as well and recompute just those pixels - O(m) instead of
    O(m·n) for the energy part. The DP and the copy are still O(m·n), so the
    total stays O((k_w + k_h)·m·n) but with a smaller constant. The default is
    the simple full recomputation, exactly as the assignment describes.
"""

from typing import Callable

import numpy as np

from seamcarve.dp import find_vertical_seam
from seamcarve.energy import LOCAL_ENERGY, dual_gradient_energy
from seamcarve.graphcut import find_vertical_seam_graphcut
from seamcarve.greedy import (
    find_vertical_seam_greedy,
    find_vertical_seam_greedy_lookahead,
    find_vertical_seam_greedy_multistart,
)

# Type of a seam finder: takes an (m, n) energy map, returns m column indices.
# Any algorithm with this signature can be plugged into resize().
SeamFinder = Callable[[np.ndarray], np.ndarray]
EnergyFunction = Callable[[np.ndarray], np.ndarray]

# Name -> seam finder, used by the command-line tools (--finder).
SEAM_FINDERS = {
    "dp": find_vertical_seam,                 # Question 1: optimal seam
    "greedy": find_vertical_seam_greedy,      # Question 3: local row-by-row choice
    # Question 3, alternative greedy designs:
    "greedy-multistart": find_vertical_seam_greedy_multistart,   # greedy from every start
    "greedy-lookahead": find_vertical_seam_greedy_lookahead,     # 3-row lookahead
    "graphcut": find_vertical_seam_graphcut,  # min s-t cut / max-flow: optimal, like DP
}


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
    incremental: bool = False,
) -> np.ndarray:
    """Shrink an image to (new_height, new_width) by seam carving.

    Args:
        img:         (m, n) or (m, n, c) image, e.g. uint8 RGB.
        new_height:  target number of rows,    1 <= new_height <= m.
        new_width:   target number of columns, 1 <= new_width  <= n.
        finder:      returns a minimum vertical seam for an energy map
                     (default: the dynamic-programming version).
        energy_fn:   computes the energy map of an image
                     (default: dual-gradient energy).
        incremental: False (default) recomputes the whole energy map after
                     every removal. True keeps the energy map, removes the
                     seam from it too, and recomputes only the pixels next to
                     the seam (see LOCAL_ENERGY in energy.py). Both give
                     exactly the same result; True is faster.

    Returns:
        New array of shape (new_height, new_width[, c]) with the same dtype.
    """
    height, width = img.shape[:2]
    if not (1 <= new_height <= height and 1 <= new_width <= width):
        raise ValueError(
            f"target size must be between 1x1 and the current size {height}x{width}, "
            f"got {new_height}x{new_width} (this implementation only shrinks images)"
        )
    if incremental and energy_fn not in LOCAL_ENERGY:
        raise ValueError(f"no local update rule for energy function {energy_fn.__name__}")

    out = img
    # With incremental updates the energy map is carried along with the image;
    # otherwise it is recomputed from scratch inside every step.
    energy = energy_fn(img) if incremental else None
    next_is_vertical = True
    while out.shape[1] > new_width or out.shape[0] > new_height:
        need_vertical = out.shape[1] > new_width
        need_horizontal = out.shape[0] > new_height

        # Alternate while both are needed; otherwise do whichever is left.
        if need_vertical and (next_is_vertical or not need_horizontal):
            out, energy = _remove_one_vertical(out, energy, finder, energy_fn)
        else:
            out, energy = _remove_one_horizontal(out, energy, finder, energy_fn)
        next_is_vertical = not next_is_vertical

    return np.ascontiguousarray(out)


def _remove_one_vertical(img, energy, finder, energy_fn):
    """One iteration: energy -> minimum seam -> remove. Returns (img, energy).

    energy is None in the default (full recomputation) mode.
    """
    if energy is None:
        seam = finder(energy_fn(img))            # 1. full energy, 2. seam
        return remove_vertical_seam(img, seam), None   # 3. delete it

    seam = finder(energy)                        # energy is already up to date
    img = remove_vertical_seam(img, seam)
    energy = remove_vertical_seam(energy, seam)  # drop the seam from the map too
    update_energy_near_seam(img, energy, seam, energy_fn)
    return img, energy


def _remove_one_horizontal(img, energy, finder, energy_fn):
    # Work on the transpose so the same vertical code can be reused:
    # a horizontal seam of img is a vertical seam of img transposed.
    # (The energy of a transposed image is the transposed energy, so the
    # carried energy map can be transposed the same way.)
    img_t = np.swapaxes(img, 0, 1)
    energy_t = None if energy is None else energy.T
    img_t, energy_t = _remove_one_vertical(img_t, energy_t, finder, energy_fn)
    return np.swapaxes(img_t, 0, 1), (None if energy_t is None else energy_t.T)


def update_energy_near_seam(img, energy, seam, energy_fn) -> None:
    """Recompute, in place, the energies that removing `seam` may have changed.

    Args:
        img:    the image AFTER the vertical seam was removed, shape (m, n-1[, c]).
        energy: the old energy map with the same seam removed, shape (m, n-1).
        seam:   the removed seam (column indices in the OLD image).

    Only pixels in columns seam[i]+lo .. seam[i]+hi of each row are recomputed,
    where (lo, hi) comes from LOCAL_ENERGY. Cost O(m) instead of O(m·n).
    """
    at_fn, (lo, hi) = LOCAL_ENERGY[energy_fn]
    m, n = energy.shape
    offsets = np.arange(lo, hi + 1)
    rows = np.repeat(np.arange(m), offsets.size)
    cols = (np.asarray(seam)[:, np.newaxis] + offsets).ravel()
    inside = (cols >= 0) & (cols < n)
    rows, cols = rows[inside], cols[inside]
    energy[rows, cols] = at_fn(img, rows, cols)
