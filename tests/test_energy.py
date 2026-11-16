"""Tests for step 1a: the dual-gradient energy map."""

import numpy as np
import pytest

from seamcarve.energy import dual_gradient_energy


def brute_force_energy(img: np.ndarray) -> np.ndarray:
    """Straightforward per-pixel version of the formula, used as a reference.

    Uses explicit loops and clamps neighbour indices at the border, which is
    the same as edge replication.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = img.astype(np.float64)
    m, n, c = img.shape
    out = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            left, right = img[i, max(j - 1, 0)], img[i, min(j + 1, n - 1)]
            up, down = img[max(i - 1, 0), j], img[min(i + 1, m - 1), j]
            out[i, j] = np.sum((right - left) ** 2) + np.sum((down - up) ** 2)
    return out


def test_shape_and_dtype():
    img = np.zeros((4, 7, 3), dtype=np.uint8)
    e = dual_gradient_energy(img)
    assert e.shape == (4, 7)
    assert e.dtype == np.float64


def test_constant_image_has_zero_energy():
    img = np.full((5, 6, 3), 123, dtype=np.uint8)
    assert np.all(dual_gradient_energy(img) == 0)


def test_vertical_edge_hand_computed():
    # Columns: 0, 0, 255, 255 in every channel and every row.
    # Δy = 0 everywhere (all rows equal).
    # Δx per channel: col0 = I[1]-I[0](replicated) = 0
    #                 col1 = I[2]-I[0] = 255
    #                 col2 = I[3]-I[1] = 255
    #                 col3 = I[3](replicated)-I[2] = 0
    # Energy = 3 channels * 255^2 at the two middle columns.
    row = np.array([0, 0, 255, 255], dtype=np.uint8)
    img = np.repeat(np.tile(row, (3, 1))[:, :, None], 3, axis=2)
    expected_row = np.array([0, 3 * 255**2, 3 * 255**2, 0], dtype=np.float64)
    np.testing.assert_array_equal(dual_gradient_energy(img), np.tile(expected_row, (3, 1)))


def test_horizontal_edge_is_transpose_of_vertical_edge():
    row = np.array([0, 0, 255, 255], dtype=np.uint8)
    img = np.repeat(np.tile(row, (3, 1))[:, :, None], 3, axis=2)
    img_t = img.transpose(1, 0, 2)
    np.testing.assert_array_equal(dual_gradient_energy(img_t), dual_gradient_energy(img).T)


def test_no_uint8_wraparound():
    # 0 next to 255: if the code subtracted uint8 values it would get 1, not 255.
    img = np.array([[0, 255]], dtype=np.uint8)            # 1 x 2 grayscale
    # col0: right - left(replicated) = 255 - 0 = 255 ; col1: 255 - 0 = 255
    np.testing.assert_array_equal(dual_gradient_energy(img), [[255.0**2, 255.0**2]])


def test_grayscale_input():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(5, 4), dtype=np.uint8)
    np.testing.assert_allclose(dual_gradient_energy(gray), brute_force_energy(gray))


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 5, 3), (5, 1, 3), (2, 2, 3), (6, 9, 3)])
def test_matches_brute_force(shape):
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    np.testing.assert_allclose(dual_gradient_energy(img), brute_force_energy(img))


def test_energy_is_non_negative():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    assert np.all(dual_gradient_energy(img) >= 0)


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        dual_gradient_energy(np.zeros(5))


# ---------------------------------------------------------------- Sobel energy

from seamcarve.energy import ENERGY_FUNCTIONS, sobel_energy  # noqa: E402

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def brute_force_sobel(img: np.ndarray) -> np.ndarray:
    """Explicit 3x3 convolution loop with clamped (edge-replicated) borders."""
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = img.astype(np.float64)
    m, n, _ = img.shape
    out = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            gx = gy = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    pixel = img[min(max(i + di, 0), m - 1), min(max(j + dj, 0), n - 1)]
                    gx = gx + SOBEL_X[di + 1, dj + 1] * pixel
                    gy = gy + SOBEL_Y[di + 1, dj + 1] * pixel
            out[i, j] = np.sum(gx * gx) + np.sum(gy * gy)
    return out


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 6, 3), (6, 1, 3), (3, 3, 3), (7, 9, 3), (5, 4)])
def test_sobel_matches_brute_force(shape):
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    np.testing.assert_allclose(sobel_energy(img), brute_force_sobel(img))


def test_sobel_constant_image_has_zero_energy():
    assert np.all(sobel_energy(np.full((4, 5, 3), 77, dtype=np.uint8)) == 0)


def test_sobel_vertical_edge_hand_computed():
    # Columns 0, 0, 255, 255; every row identical, so Gy = 0.
    # Gx = (1 + 2 + 1) * (right - left) = 4 * 255 at the two middle columns.
    row = np.array([0, 0, 255, 255], dtype=np.uint8)
    img = np.tile(row, (3, 1))                         # grayscale, 1 channel
    expected_row = np.array([0, (4 * 255) ** 2, (4 * 255) ** 2, 0], dtype=np.float64)
    np.testing.assert_array_equal(sobel_energy(img), np.tile(expected_row, (3, 1)))


@pytest.mark.parametrize("energy_fn", list(ENERGY_FUNCTIONS.values()))
def test_energy_of_transpose_is_transpose_of_energy(energy_fn):
    # resize() removes horizontal seams by transposing the image, so this
    # symmetry must hold for every energy function.
    rng = np.random.default_rng(12)
    img = rng.integers(0, 256, size=(6, 9, 3), dtype=np.uint8)
    np.testing.assert_array_equal(energy_fn(img.transpose(1, 0, 2)), energy_fn(img).T)


def test_sobel_smooths_single_pixel_noise_relative_to_edges():
    # A single bright pixel vs a full vertical edge of the same contrast:
    # Sobel spreads the lone pixel's response, so the edge stands out more
    # than with dual-gradient. Ratio edge/noise is larger for Sobel.
    noise = np.zeros((7, 7)); noise[3, 3] = 255
    edge = np.zeros((7, 7)); edge[:, 4:] = 255
    ratio_dual = dual_gradient_energy(edge).max() / dual_gradient_energy(noise).max()
    ratio_sobel = sobel_energy(edge).max() / sobel_energy(noise).max()
    assert ratio_sobel > ratio_dual


def test_sobel_rejects_bad_shape():
    with pytest.raises(ValueError):
        sobel_energy(np.zeros(5))
