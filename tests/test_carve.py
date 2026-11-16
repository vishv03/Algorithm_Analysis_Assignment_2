"""Tests for step 1c: seam removal and the resize loop."""

import numpy as np
import pytest

from seamcarve.carve import remove_horizontal_seam, remove_vertical_seam, resize
from seamcarve.dp import find_horizontal_seam
from seamcarve.energy import dual_gradient_energy


def unique_image(m: int, n: int, c: int = 3) -> np.ndarray:
    """Image where every pixel is different, so we can tell exactly which were removed."""
    return np.arange(m * n * c, dtype=np.int64).reshape(m, n, c)


def is_subsequence(short, long) -> bool:
    """True if `short` is `long` with some elements deleted (order kept)."""
    it = iter(long.tolist())
    return all(any(x == y for y in it) for x in short.tolist())


# ---------------------------------------------------------------- remove_vertical_seam

def test_remove_vertical_removes_exactly_the_seam_pixels():
    img = unique_image(4, 5)
    seam = np.array([0, 1, 2, 2])
    out = remove_vertical_seam(img, seam)
    assert out.shape == (4, 4, 3)
    for i in range(4):
        expected_row = np.delete(img[i], seam[i], axis=0)
        np.testing.assert_array_equal(out[i], expected_row)


def test_remove_vertical_first_and_last_columns():
    img = unique_image(3, 4)
    np.testing.assert_array_equal(remove_vertical_seam(img, [0, 0, 0]), img[:, 1:])
    np.testing.assert_array_equal(remove_vertical_seam(img, [3, 3, 3]), img[:, :-1])


def test_remove_vertical_grayscale():
    img = np.arange(12).reshape(3, 4)
    out = remove_vertical_seam(img, [1, 2, 3])
    np.testing.assert_array_equal(out, [[0, 2, 3], [4, 5, 7], [8, 9, 10]])


def test_remove_vertical_does_not_modify_input():
    img = unique_image(3, 3)
    copy = img.copy()
    remove_vertical_seam(img, [1, 1, 1])
    np.testing.assert_array_equal(img, copy)


@pytest.mark.parametrize("seam", [[0, 0], [0, 0, 0, 0], [0, 5, 0], [-1, 0, 0]])
def test_remove_vertical_rejects_bad_seam(seam):
    with pytest.raises(ValueError):
        remove_vertical_seam(unique_image(3, 4), seam)


def test_cannot_remove_from_one_pixel_wide_image():
    with pytest.raises(ValueError):
        remove_vertical_seam(unique_image(3, 1), [0, 0, 0])


# ---------------------------------------------------------------- remove_horizontal_seam

def test_remove_horizontal_removes_exactly_the_seam_pixels():
    img = unique_image(5, 4)
    seam = np.array([0, 1, 2, 2])          # one row index per column
    out = remove_horizontal_seam(img, seam)
    assert out.shape == (4, 4, 3)
    for j in range(4):
        expected_col = np.delete(img[:, j], seam[j], axis=0)
        np.testing.assert_array_equal(out[:, j], expected_col)


def test_remove_horizontal_matches_transposed_vertical():
    img = unique_image(5, 6)
    seam = np.array([4, 3, 3, 2, 1, 0])
    via_horizontal = remove_horizontal_seam(img, seam)
    via_transpose = remove_vertical_seam(img.transpose(1, 0, 2), seam).transpose(1, 0, 2)
    np.testing.assert_array_equal(via_horizontal, via_transpose)


# ---------------------------------------------------------------- resize

@pytest.mark.parametrize("target", [(10, 12), (10, 7), (6, 12), (6, 7), (1, 1), (1, 12), (10, 1)])
def test_resize_reaches_target_shape_and_keeps_dtype(target):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(10, 12, 3), dtype=np.uint8)
    out = resize(img, *target)
    assert out.shape == (*target, 3)
    assert out.dtype == np.uint8


def test_resize_to_same_size_is_identity():
    img = unique_image(5, 6)
    np.testing.assert_array_equal(resize(img, 5, 6), img)


@pytest.mark.parametrize("target", [(11, 12), (10, 13), (0, 5), (5, 0), (-1, 5)])
def test_resize_rejects_invalid_targets(target):
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize(img, *target)


def test_resize_does_not_modify_input():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(8, 9, 3), dtype=np.uint8)
    copy = img.copy()
    resize(img, 5, 6)
    np.testing.assert_array_equal(img, copy)


def test_width_only_resize_only_deletes_pixels_within_rows():
    # Every output row must be the original row with some pixels deleted,
    # in the same order (seam carving never moves or invents pixels).
    rng = np.random.default_rng(2)
    img = rng.permutation(8 * 10).reshape(8, 10)      # unique grayscale values
    out = resize(img, 8, 6)
    assert out.shape == (8, 6)
    for i in range(8):
        assert is_subsequence(out[i], img[i])


def test_height_only_resize_only_deletes_pixels_within_columns():
    rng = np.random.default_rng(3)
    img = rng.permutation(10 * 8).reshape(10, 8)
    out = resize(img, 6, 8)
    for j in range(8):
        assert is_subsequence(out[:, j], img[:, j])


def test_single_horizontal_step_uses_the_dp_horizontal_seam():
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, size=(6, 7, 3), dtype=np.uint8)
    seam = find_horizontal_seam(dual_gradient_energy(img))
    np.testing.assert_array_equal(resize(img, 5, 7), remove_horizontal_seam(img, seam))


def test_flat_background_is_removed_before_the_object():
    # Black 6x20 image with a 3-pixel-wide TEXTURED object in the middle
    # (random bright colours, so every object pixel has high gradient energy).
    # Shrinking to width 10 should remove black background only, so the
    # object must survive untouched.
    # Note: a single flat-colour object would NOT work - its interior has
    # zero gradient (255 - 255 = 0), the same energy as the background.
    rng = np.random.default_rng(5)
    img = np.zeros((6, 20, 3), dtype=np.uint8)
    obj = rng.integers(100, 256, size=(6, 3, 3), dtype=np.uint8)
    img[:, 9:12] = obj
    out = resize(img, 6, 10)
    for i in range(6):
        row = out[i]
        start = np.flatnonzero(row.any(axis=1))[0]    # first non-black pixel
        np.testing.assert_array_equal(row[start:start + 3], obj[i])


def test_uniform_object_interior_has_zero_energy():
    # Documents the limitation above: the middle of a flat white bar is as
    # "cheap" as flat black background under gradient energy.
    img = np.zeros((4, 9, 3), dtype=np.uint8)
    img[:, 3:6] = 255
    e = dual_gradient_energy(img)
    assert np.all(e[:, 4] == 0) and np.all(e[:, 0] == 0)
    assert np.all(e[:, 3] > 0)                         # the bar's edges are expensive


def test_custom_finder_is_used():
    # A dummy finder that always removes column 0 turns resize into cropping
    # from the left - proving resize really calls the finder it is given.
    def leftmost(energy):
        return np.zeros(energy.shape[0], dtype=int)

    img = unique_image(4, 6)
    np.testing.assert_array_equal(resize(img, 4, 3, finder=leftmost), img[:, 3:])
    np.testing.assert_array_equal(resize(img, 2, 6, finder=leftmost), img[2:, :])
