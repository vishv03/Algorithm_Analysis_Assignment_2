"""Tests for the two improvements: low-memory DP and local energy update.

Both must give EXACTLY the same results as the simple versions.
"""

import tracemalloc

import numpy as np
import pytest

from seamcarve.carve import remove_vertical_seam, resize, update_energy_near_seam
from seamcarve.dp import (
    cumulative_energy_vertical,
    find_vertical_seam,
    find_vertical_seam_low_memory,
    min_vertical_seam_cost,
)
from seamcarve.energy import (
    LOCAL_ENERGY,
    dual_gradient_energy,
    dual_gradient_energy_at,
    sobel_energy,
    sobel_energy_at,
)

ENERGY_PAIRS = [
    (dual_gradient_energy, dual_gradient_energy_at),
    (sobel_energy, sobel_energy_at),
]


def random_seam(rng, m, n):
    """A random valid vertical seam, biased to touch the borders sometimes."""
    seam = [int(rng.integers(0, n))]
    for _ in range(m - 1):
        seam.append(int(np.clip(seam[-1] + rng.integers(-1, 2), 0, n - 1)))
    return np.array(seam)


# ---------------------------------------------------------------- low-memory DP

@pytest.mark.parametrize("m,n", [(1, 1), (1, 7), (7, 1), (5, 5), (20, 30), (31, 9)])
@pytest.mark.parametrize("seed", range(5))
def test_low_memory_seam_identical_to_full_table(m, n, seed):
    rng = np.random.default_rng(seed)
    energy = rng.random((m, n))
    np.testing.assert_array_equal(find_vertical_seam_low_memory(energy), find_vertical_seam(energy))


@pytest.mark.parametrize("seed", range(10))
def test_low_memory_identical_with_ties(seed):
    rng = np.random.default_rng(50 + seed)
    energy = rng.integers(0, 3, size=(12, 10)).astype(np.float64)
    np.testing.assert_array_equal(find_vertical_seam_low_memory(energy), find_vertical_seam(energy))


@pytest.mark.parametrize("seed", range(5))
def test_min_cost_two_rows_matches_table(seed):
    energy = np.random.default_rng(seed).random((15, 12))
    M, _ = cumulative_energy_vertical(energy)
    assert min_vertical_seam_cost(energy) == M[-1].min()


def peak_bytes(fn, *args):
    tracemalloc.start()
    fn(*args)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def test_low_memory_uses_much_less_memory():
    energy = np.random.default_rng(0).random((300, 400))
    full = peak_bytes(find_vertical_seam, energy)
    low = peak_bytes(find_vertical_seam_low_memory, energy)
    # Full table: about 9 bytes per pixel; low-memory: about 1 byte per pixel.
    assert low * 4 < full


# ---------------------------------------------------------------- energy at selected pixels

@pytest.mark.parametrize("full_fn,at_fn", ENERGY_PAIRS)
@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 6, 3), (6, 1, 3), (7, 9, 3), (6, 5)])
def test_energy_at_matches_full_map(full_fn, at_fn, shape):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    rows, cols = np.indices(shape[:2])
    np.testing.assert_array_equal(at_fn(img, rows.ravel(), cols.ravel()), full_fn(img).ravel())


# ---------------------------------------------------------------- local energy update

@pytest.mark.parametrize("full_fn", [dual_gradient_energy, sobel_energy])
@pytest.mark.parametrize("seed", range(10))
def test_local_update_equals_full_recompute_after_every_removal(full_fn, seed):
    # Remove many random seams (including ones along the borders); after each
    # removal the locally updated map must equal a full recomputation exactly.
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(9, 14, 3), dtype=np.uint8)
    energy = full_fn(img)
    while img.shape[1] > 1:
        seam = random_seam(rng, img.shape[0], img.shape[1])
        img = remove_vertical_seam(img, seam)
        energy = remove_vertical_seam(energy, seam)
        update_energy_near_seam(img, energy, seam, full_fn)
        np.testing.assert_array_equal(energy, full_fn(img))


@pytest.mark.parametrize("full_fn", [dual_gradient_energy, sobel_energy])
def test_local_update_window_is_needed(full_fn):
    # Sanity check that the test above has teeth: WITHOUT the update the
    # carried energy map goes wrong.
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(9, 14, 3), dtype=np.uint8)
    seam = random_seam(rng, 9, 14)
    stale = remove_vertical_seam(full_fn(img), seam)
    assert not np.array_equal(stale, full_fn(remove_vertical_seam(img, seam)))


@pytest.mark.parametrize("energy_fn", list(LOCAL_ENERGY))
@pytest.mark.parametrize("target", [(20, 15), (14, 25), (12, 16), (1, 1), (20, 1)])
def test_incremental_resize_identical_to_full_resize(energy_fn, target):
    rng = np.random.default_rng(9)
    img = rng.integers(0, 256, size=(20, 25, 3), dtype=np.uint8)
    slow = resize(img, *target, energy_fn=energy_fn)
    fast = resize(img, *target, energy_fn=energy_fn, incremental=True)
    np.testing.assert_array_equal(fast, slow)


def test_incremental_with_low_memory_finder_identical():
    rng = np.random.default_rng(10)
    img = rng.integers(0, 256, size=(18, 22, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        resize(img, 12, 14, finder=find_vertical_seam_low_memory, incremental=True),
        resize(img, 12, 14),
    )


def test_incremental_rejects_energy_without_local_rule():
    def custom_energy(img):
        return dual_gradient_energy(img)

    with pytest.raises(ValueError):
        resize(np.zeros((5, 5, 3), dtype=np.uint8), 4, 4, energy_fn=custom_energy, incremental=True)
