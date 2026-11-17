"""Step 1a: the energy map.

The energy of a pixel measures how "important" it is, i.e. how much the image
would be disrupted if that pixel were removed. Pixels in flat regions (sky,
grass) have low energy; pixels on edges and in detailed regions have high energy.

Two energy functions are provided (alternative designs, same interface):
    dual_gradient_energy  - central differences, 2 neighbours per direction (default)
    sobel_energy          - Sobel filter, a 3x3 weighted neighbourhood

1) Dual-gradient energy:

    Δx(i, j) = I(i, j+1) - I(i, j-1)          (horizontal central difference)
    Δy(i, j) = I(i+1, j) - I(i-1, j)          (vertical central difference)

    e(i, j)  = Σ over channels c in {R, G, B} of  Δx_c(i, j)² + Δy_c(i, j)²

Border pixels have a missing neighbour. We handle this by "edge replication":
the image is padded by one pixel on every side, copying the nearest edge pixel
outward. So for the left-most column, I(i, -1) is treated as I(i, 0).
This keeps one formula for every pixel and needs no special cases.

Time:  O(m·n) for an m x n image (a constant amount of work per pixel).
Space: O(m·n) for the padded copy and the output map.
"""

import numpy as np


def dual_gradient_energy(img: np.ndarray) -> np.ndarray:
    """Compute the dual-gradient energy map of an image.

    Args:
        img: array of shape (m, n, c) for a colour image (c = 3 for RGB) or
             (m, n) for a grayscale image. Any numeric dtype (e.g. uint8).

    Returns:
        float64 array of shape (m, n): the energy of every pixel (always >= 0).
    """
    if img.ndim == 2:                      # grayscale -> treat as 1 channel
        img = img[:, :, np.newaxis]
    if img.ndim != 3:
        raise ValueError(f"expected an (m, n) or (m, n, c) image, got shape {img.shape}")

    # Convert to float FIRST. With uint8, 0 - 255 would wrap around to 1
    # instead of giving -255, which would silently corrupt the energy.
    img = img.astype(np.float64)

    # Pad 1 pixel on top/bottom/left/right (not along the channel axis),
    # repeating the edge values. padded has shape (m+2, n+2, c), and the
    # original pixel (i, j) sits at padded[i+1, j+1].
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")

    # Central differences, computed for all pixels at once by slicing:
    #   right neighbour  of (i, j) = padded[i+1, j+2]  -> padded[1:-1, 2:]
    #   left  neighbour  of (i, j) = padded[i+1, j]    -> padded[1:-1, :-2]
    #   lower neighbour  of (i, j) = padded[i+2, j+1]  -> padded[2:, 1:-1]
    #   upper neighbour  of (i, j) = padded[i,   j+1]  -> padded[:-2, 1:-1]
    dx = padded[1:-1, 2:] - padded[1:-1, :-2]
    dy = padded[2:, 1:-1] - padded[:-2, 1:-1]

    # Square, add, and sum over the colour channels -> shape (m, n).
    return np.sum(dx * dx + dy * dy, axis=2)


def sobel_energy(img: np.ndarray) -> np.ndarray:
    """Compute the Sobel energy map of an image.

    The Sobel filter estimates the gradient with two 3x3 kernels, which also
    look at the diagonal neighbours and give the centre row/column weight 2:

              [-1  0  +1]              [-1  -2  -1]
        Gx =  [-2  0  +2]        Gy =  [ 0   0   0]
              [-1  0  +1]              [+1  +2  +1]

        e(i, j) = Σ over channels of  Gx(i, j)² + Gy(i, j)²

    Compared with dual-gradient, the extra smoothing makes Sobel less
    sensitive to single-pixel noise, at about 3x the arithmetic per pixel.
    Borders use the same edge replication. Time and space are still O(m·n).

    Args / Returns: same as dual_gradient_energy.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.ndim != 3:
        raise ValueError(f"expected an (m, n) or (m, n, c) image, got shape {img.shape}")

    p = np.pad(img.astype(np.float64), ((1, 1), (1, 1), (0, 0)), mode="edge")

    # Each slice is one position of the 3x3 window, for all pixels at once.
    # p[a:b, c:d] with rows offset r in {0,1,2} and columns offset c in {0,1,2}:
    top_left, top, top_right = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    left, right = p[1:-1, :-2], p[1:-1, 2:]
    bottom_left, bottom, bottom_right = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]

    # Gx: right column minus left column, centre row weighted 2.
    gx = (top_right + 2 * right + bottom_right) - (top_left + 2 * left + bottom_left)
    # Gy: bottom row minus top row, centre column weighted 2.
    gy = (bottom_left + 2 * bottom + bottom_right) - (top_left + 2 * top + top_right)

    return np.sum(gx * gx + gy * gy, axis=2)


def forward_energy(img: np.ndarray) -> np.ndarray:
    """Simple forward energy: the new edge created if pixel (i, j) is removed.

    Removing (i, j) makes its left and right neighbours adjacent, so the cost is
    how different they are:

        e(i, j) = Σ over channels of |I(i, j+1) - I(i, j-1)|

    It measures the energy INSERTED by the removal rather than the energy
    removed. Used by the graph cut comparison (graphcut_compare.py) as the
    brief suggests. (Full forward energy, Rubinstein et al. 2008, also charges
    the new vertical neighbours; that needs costs on the diagonal graph edges
    and is not implemented here.) Borders use edge replication. O(m·n).
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.ndim != 3:
        raise ValueError(f"expected an (m, n) or (m, n, c) image, got shape {img.shape}")
    p = np.pad(img.astype(np.float64), ((0, 0), (1, 1), (0, 0)), mode="edge")
    return np.sum(np.abs(p[:, 2:] - p[:, :-2]), axis=2)


# Name -> function, used by the command-line tools (--energy).
ENERGY_FUNCTIONS = {
    "dual-gradient": dual_gradient_energy,
    "sobel": sobel_energy,
    "forward": forward_energy,
}


# ---------------------------------------------------------------------------
# Improvement: local energy update.
#
# After a vertical seam is removed, a pixel's energy can only change if one of
# the neighbours its formula reads has changed. Those pixels lie in a small
# window around the seam in each row, so instead of recomputing all m·n
# energies we recompute only about m·w of them (w = window width) - O(m)
# instead of O(m·n) for this part. The functions below compute the energy of
# selected pixels only; resize() decides which pixels to pass in.
#
# Borders: clamping an index to [0, size-1] is the same as edge replication.
# ---------------------------------------------------------------------------

def _pixels(img: np.ndarray, rows: np.ndarray, cols: np.ndarray, di: int, dj: int) -> np.ndarray:
    """Values of the pixels at (rows+di, cols+dj), clamped to the image, as float64."""
    m, n = img.shape[:2]
    r = np.clip(rows + di, 0, m - 1)
    c = np.clip(cols + dj, 0, n - 1)
    values = img[r, c].astype(np.float64)
    return values if values.ndim == 2 else values[:, np.newaxis]   # (k, channels)


def dual_gradient_energy_at(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Dual-gradient energy of the pixels (rows[k], cols[k]) only. Returns length k."""
    dx = _pixels(img, rows, cols, 0, +1) - _pixels(img, rows, cols, 0, -1)
    dy = _pixels(img, rows, cols, +1, 0) - _pixels(img, rows, cols, -1, 0)
    return np.sum(dx * dx + dy * dy, axis=1)


def sobel_energy_at(img: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sobel energy of the pixels (rows[k], cols[k]) only. Returns length k."""
    def p(di, dj):
        return _pixels(img, rows, cols, di, dj)

    gx = (p(-1, 1) + 2 * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2 * p(0, -1) + p(1, -1))
    gy = (p(1, -1) + 2 * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2 * p(-1, 0) + p(-1, 1))
    return np.sum(gx * gx + gy * gy, axis=1)


# For each energy function: its "at selected pixels" version, and the column
# offsets around the removed seam (in the NEW, narrower image) whose energy
# can change. Pixel j of row i is recomputed if seam[i] + lo <= j <= seam[i] + hi.
#
#   dual-gradient reads its left/right and up/down neighbours. Only the two
#   pixels that become adjacent across the gap (new columns seam[i]-1 and
#   seam[i]) change, because the seam moves at most 1 column per row: (-1, 0).
#   Sobel reads a 3x3 window, so changes in rows i-1 and i+1 (whose seam pixel
#   is within 1 column) reach 1 further on each side: (-3, 2) is safe.
LOCAL_ENERGY = {
    dual_gradient_energy: (dual_gradient_energy_at, (-1, 0)),
    sobel_energy: (sobel_energy_at, (-3, 2)),
}
