"""Step 1a: the energy map.

The energy of a pixel measures how "important" it is, i.e. how much the image
would be disrupted if that pixel were removed. Pixels in flat regions (sky,
grass) have low energy; pixels on edges and in detailed regions have high energy.

We use the dual-gradient energy function:

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
