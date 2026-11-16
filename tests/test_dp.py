"""Tests for step 1b: the DP minimum seam search.

The key test is the brute-force comparison: on small grids we list EVERY
valid seam, take the cheapest one, and check the DP finds the same cost.
"""

import numpy as np
import pytest

from seamcarve.dp import (
    backtrack_vertical,
    cumulative_energy_vertical,
    find_horizontal_seam,
    find_vertical_seam,
)


# ---------------------------------------------------------------- helpers

def all_vertical_seams(m: int, n: int):
    """Yield every valid vertical seam of an m x n grid (exponentially many)."""
    def extend(seam):
        if len(seam) == m:
            yield list(seam)
            return
        last = seam[-1]
        for j in (last - 1, last, last + 1):
            if 0 <= j < n:
                yield from extend(seam + [j])

    for start in range(n):
        yield from extend([start])


def vertical_cost(energy: np.ndarray, seam) -> float:
    return float(energy[np.arange(energy.shape[0]), seam].sum())


def horizontal_cost(energy: np.ndarray, seam) -> float:
    return float(energy[seam, np.arange(energy.shape[1])].sum())


def assert_valid_vertical(seam, m: int, n: int):
    seam = np.asarray(seam)
    assert seam.shape == (m,)
    assert np.all((seam >= 0) & (seam < n)), "seam leaves the image"
    assert np.all(np.abs(np.diff(seam)) <= 1), "seam is not connected"


# ---------------------------------------------------------------- hand-worked example

ENERGY = np.array([
    [1, 4, 3, 5],
    [2, 9, 1, 6],
    [7, 3, 8, 2],
    [4, 6, 5, 1],
], dtype=np.float64)

#   M row 0 = [ 1,  4,  3,  5]
#   M row 1 = [ 2+1, 9+1, 1+3, 6+3] = [ 3, 10,  4,  9]
#   M row 2 = [ 7+3, 3+3, 8+4, 2+4] = [10,  6, 12,  6]
#   M row 3 = [ 4+6, 6+6, 5+6, 1+6] = [10, 12, 11,  7]
#   min of last row is 7 at column 3 -> seam (bottom-up) 3, 3, 2, 2
EXPECTED_M = np.array([
    [1, 4, 3, 5],
    [3, 10, 4, 9],
    [10, 6, 12, 6],
    [10, 12, 11, 7],
], dtype=np.float64)


def test_hand_worked_table():
    M, back = cumulative_energy_vertical(ENERGY)
    np.testing.assert_array_equal(M, EXPECTED_M)
    assert back.dtype == np.int8
    assert set(np.unique(back)) <= {-1, 0, 1}


def test_hand_worked_seam():
    seam = find_vertical_seam(ENERGY)
    np.testing.assert_array_equal(seam, [2, 2, 3, 3])
    assert vertical_cost(ENERGY, seam) == 7


def test_backtrack_from_tables():
    M, back = cumulative_energy_vertical(ENERGY)
    np.testing.assert_array_equal(backtrack_vertical(M, back), [2, 2, 3, 3])


# ---------------------------------------------------------------- brute force

@pytest.mark.parametrize("m,n", [(1, 1), (1, 4), (4, 1), (2, 2), (3, 3), (4, 5), (6, 5), (6, 2)])
@pytest.mark.parametrize("seed", range(10))
def test_vertical_matches_brute_force(m, n, seed):
    rng = np.random.default_rng(seed)
    energy = rng.random((m, n)) * 100
    best = min(vertical_cost(energy, s) for s in all_vertical_seams(m, n))

    seam = find_vertical_seam(energy)
    assert_valid_vertical(seam, m, n)
    assert vertical_cost(energy, seam) == pytest.approx(best)


@pytest.mark.parametrize("seed", range(20))
def test_vertical_matches_brute_force_with_ties(seed):
    # Small integer energies produce many equal-cost seams.
    rng = np.random.default_rng(100 + seed)
    m, n = 5, 4
    energy = rng.integers(0, 3, size=(m, n)).astype(np.float64)
    best = min(vertical_cost(energy, s) for s in all_vertical_seams(m, n))

    seam = find_vertical_seam(energy)
    assert_valid_vertical(seam, m, n)
    assert vertical_cost(energy, seam) == best


def test_dp_table_last_row_matches_brute_force_per_end_column():
    # M[m-1][j] must be the cheapest seam that ENDS at column j.
    rng = np.random.default_rng(7)
    m, n = 5, 4
    energy = rng.random((m, n))
    M, _ = cumulative_energy_vertical(energy)
    for j in range(n):
        best_j = min(vertical_cost(energy, s) for s in all_vertical_seams(m, n) if s[-1] == j)
        assert M[-1, j] == pytest.approx(best_j)


@pytest.mark.parametrize("m,n", [(3, 3), (4, 6), (5, 6), (2, 5)])
@pytest.mark.parametrize("seed", range(5))
def test_horizontal_matches_brute_force(m, n, seed):
    rng = np.random.default_rng(seed)
    energy = rng.random((m, n))
    # A horizontal seam of an m x n grid is a vertical seam of the n x m transpose.
    best = min(horizontal_cost(energy, s) for s in all_vertical_seams(n, m))

    seam = find_horizontal_seam(energy)
    assert_valid_vertical(seam, n, m)          # n entries, each a row index < m
    assert horizontal_cost(energy, seam) == pytest.approx(best)


def test_horizontal_is_vertical_of_transpose():
    rng = np.random.default_rng(3)
    energy = rng.random((7, 9))
    np.testing.assert_array_equal(find_horizontal_seam(energy), find_vertical_seam(energy.T))


# ---------------------------------------------------------------- edge cases

def test_single_column_seam_is_all_zeros():
    energy = np.arange(5, dtype=np.float64).reshape(5, 1)
    np.testing.assert_array_equal(find_vertical_seam(energy), [0, 0, 0, 0, 0])


def test_single_row_picks_minimum_pixel():
    energy = np.array([[5.0, 2.0, 7.0, 1.0, 3.0]])
    np.testing.assert_array_equal(find_vertical_seam(energy), [3])


def test_constant_energy_gives_valid_seam():
    energy = np.full((6, 4), 2.5)
    seam = find_vertical_seam(energy)
    assert_valid_vertical(seam, 6, 4)
    assert vertical_cost(energy, seam) == 6 * 2.5


def test_seam_avoids_high_energy_wall():
    # A wall of high energy in column 2, except a cheap gap; the seam should
    # stay on the cheap side and never touch the wall.
    energy = np.zeros((6, 5))
    energy[:, 2] = 1000
    seam = find_vertical_seam(energy)
    assert_valid_vertical(seam, 6, 5)
    assert np.all(seam != 2)


def test_diagonal_path_is_followed():
    # Only the main diagonal is cheap; the best seam must move diagonally.
    energy = np.full((4, 4), 10.0)
    np.fill_diagonal(energy, 0.0)
    np.testing.assert_array_equal(find_vertical_seam(energy), [0, 1, 2, 3])


def test_does_not_modify_input():
    energy = np.random.default_rng(0).random((4, 4))
    copy = energy.copy()
    find_vertical_seam(energy)
    find_horizontal_seam(energy)
    np.testing.assert_array_equal(energy, copy)


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((0, 3)), np.zeros((3, 0)), np.zeros((2, 2, 3))])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        find_vertical_seam(bad)
