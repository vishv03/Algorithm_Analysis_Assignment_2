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
