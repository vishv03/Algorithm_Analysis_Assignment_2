"""Tests for Question 3: the greedy seam finder.

Greedy is not optimal, so instead of matching brute force we check that:
  * it always returns a VALID seam,
  * it really makes the local choice the assignment describes,
  * its cost is never below the DP (optimal) cost,
  * on a hand-made "trap" it is strictly worse than DP.
"""

import numpy as np
import pytest

from seamcarve.carve import SEAM_FINDERS, resize
from seamcarve.dp import find_vertical_seam
from seamcarve.greedy import find_horizontal_seam_greedy, find_vertical_seam_greedy
from tests.test_dp import ENERGY, all_vertical_seams, assert_valid_vertical, vertical_cost


# ---------------------------------------------------------------- hand-worked examples

TRAP = np.array([
    [1, 2, 2],
    [9, 9, 1],
    [9, 9, 1],
], dtype=np.float64)


def test_trap_greedy_is_much_worse_than_dp():
    # Greedy takes the cheap 1 in row 0, then column 2 is out of reach.
    greedy = find_vertical_seam_greedy(TRAP)
    np.testing.assert_array_equal(greedy, [0, 0, 0])
    assert vertical_cost(TRAP, greedy) == 19

    dp = find_vertical_seam(TRAP)
    assert vertical_cost(TRAP, dp) == 4


def test_hand_worked_example_from_dp_tests():
    # Row 0: min is 1 at column 0.
    # Row 1: candidates col 0 = 2, col 1 = 9           -> col 0
    # Row 2: candidates col 0 = 7, col 1 = 3           -> col 1
    # Row 3: candidates col 0 = 4, col 1 = 6, col 2 = 5 -> col 0
    seam = find_vertical_seam_greedy(ENERGY)
    np.testing.assert_array_equal(seam, [0, 0, 1, 0])
    assert vertical_cost(ENERGY, seam) == 10          # DP finds 7


# ---------------------------------------------------------------- properties

@pytest.mark.parametrize("m,n", [(1, 1), (1, 5), (5, 1), (2, 2), (4, 5), (30, 20), (7, 2)])
@pytest.mark.parametrize("seed", range(10))
def test_valid_seam(m, n, seed):
    energy = np.random.default_rng(seed).random((m, n))
    assert_valid_vertical(find_vertical_seam_greedy(energy), m, n)


@pytest.mark.parametrize("seed", range(20))
def test_makes_the_local_choice(seed):
    # Every step must pick the cheapest allowed pixel in the next row.
    energy = np.random.default_rng(seed).random((12, 9))
    seam = find_vertical_seam_greedy(energy)
    assert seam[0] == np.argmin(energy[0])
    for i in range(1, len(seam)):
        lo, hi = max(seam[i - 1] - 1, 0), min(seam[i - 1] + 1, energy.shape[1] - 1)
        assert energy[i, seam[i]] == energy[i, lo:hi + 1].min()


@pytest.mark.parametrize("m,n", [(3, 3), (4, 5), (6, 4)])
@pytest.mark.parametrize("seed", range(10))
def test_never_cheaper_than_optimal(m, n, seed):
    energy = np.random.default_rng(seed).random((m, n)) * 100
    best = min(vertical_cost(energy, s) for s in all_vertical_seams(m, n))
    greedy = vertical_cost(energy, find_vertical_seam_greedy(energy))
    assert greedy >= best - 1e-9
    assert vertical_cost(energy, find_vertical_seam(energy)) == pytest.approx(best)


def test_greedy_is_worse_on_average_random_maps():
    # On random maps greedy should lose to DP most of the time.
    rng = np.random.default_rng(0)
    worse = 0
    for _ in range(50):
        energy = rng.random((40, 30))
        g = vertical_cost(energy, find_vertical_seam_greedy(energy))
        d = vertical_cost(energy, find_vertical_seam(energy))
        assert g >= d - 1e-9
        worse += g > d + 1e-9
    assert worse >= 45


def test_horizontal_is_vertical_of_transpose():
    energy = np.random.default_rng(3).random((7, 9))
    np.testing.assert_array_equal(find_horizontal_seam_greedy(energy),
                                  find_vertical_seam_greedy(energy.T))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        find_vertical_seam_greedy(np.zeros(5))
    with pytest.raises(ValueError):
        find_vertical_seam_greedy(np.zeros((0, 3)))


# ---------------------------------------------------------------- inside resize()

def test_resize_with_greedy_finder():
    img = np.random.default_rng(1).integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
    out = resize(img, 15, 22, finder=SEAM_FINDERS["greedy"])
    assert out.shape == (15, 22, 3)
    assert out.dtype == np.uint8


def test_resize_greedy_incremental_matches_full():
    img = np.random.default_rng(2).integers(0, 256, size=(18, 25, 3), dtype=np.uint8)
    full = resize(img, 14, 19, finder=find_vertical_seam_greedy)
    fast = resize(img, 14, 19, finder=find_vertical_seam_greedy, incremental=True)
    np.testing.assert_array_equal(full, fast)


def test_resize_greedy_horizontal_and_grayscale():
    gray = np.random.default_rng(4).integers(0, 256, size=(20, 16), dtype=np.uint8)
    out = resize(gray, 13, 16, finder=find_vertical_seam_greedy)   # horizontal seams only
    assert out.shape == (13, 16)
    out = resize(gray, 15, 11, finder=find_vertical_seam_greedy)   # both directions
    assert out.shape == (15, 11)


# ---------------------------------------------------------------- alternative greedy designs

from seamcarve.greedy import (  # noqa: E402
    find_vertical_seam_greedy_lookahead,
    find_vertical_seam_greedy_multistart,
    lookahead_costs,
)

VARIANTS = [find_vertical_seam_greedy_multistart,
            lambda e: find_vertical_seam_greedy_lookahead(e, 2),
            find_vertical_seam_greedy_lookahead]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("m,n", [(1, 1), (1, 5), (5, 1), (3, 3), (4, 5), (6, 4)])
@pytest.mark.parametrize("seed", range(8))
def test_variants_valid_and_between_dp_and_brute_force(variant, m, n, seed):
    energy = np.random.default_rng(seed).random((m, n)) * 100
    seam = variant(energy)
    assert_valid_vertical(seam, m, n)
    best = min(vertical_cost(energy, s) for s in all_vertical_seams(m, n))
    assert vertical_cost(energy, seam) >= best - 1e-9


def test_multistart_escapes_the_start_trap():
    seam = find_vertical_seam_greedy_multistart(TRAP)
    assert vertical_cost(TRAP, seam) == 4


@pytest.mark.parametrize("seed", range(15))
def test_multistart_is_best_of_all_single_starts(seed):
    # Each start s must make the same choices as plain greedy started at s.
    energy = np.random.default_rng(seed).integers(0, 4, size=(9, 7)).astype(np.float64)
    m, n = energy.shape

    def greedy_from(s):
        seam = [s]
        for i in range(1, m):
            lo, hi = max(seam[-1] - 1, 0), min(seam[-1] + 1, n - 1)
            seam.append(lo + int(np.argmin(energy[i, lo:hi + 1])))
        return seam

    best = min(vertical_cost(energy, greedy_from(s)) for s in range(n))
    assert vertical_cost(energy, find_vertical_seam_greedy_multistart(energy)) == best
    plain = vertical_cost(energy, find_vertical_seam_greedy(energy))
    assert vertical_cost(energy, find_vertical_seam_greedy_multistart(energy)) <= plain


def test_multistart_is_still_not_optimal():
    # A trap in row 1 catches every start. The optimal seam is 1 -> 4 -> 1 -> 4
    # (columns 0, 1, 2, 2) = 10: it must take the 4 in row 1, not the cheaper 3.
    #   start col 0: 1 -> 3 -> 6 -> 8 = 18   (takes the 3, then stuck on the left)
    #   start col 1: 4 -> 3 -> 6 -> 8 = 21   (same trap)
    #   start col 2: 9 -> 4 -> 1 -> 4 = 18   (right path, but the start costs 9)
    energy = np.array([
        [1, 4, 9],
        [3, 4, 9],
        [6, 6, 1],
        [8, 8, 4],
    ], dtype=np.float64)
    seam = find_vertical_seam_greedy_multistart(energy)
    assert vertical_cost(energy, seam) == 18
    np.testing.assert_array_equal(find_vertical_seam(energy), [0, 1, 2, 2])
    assert vertical_cost(energy, find_vertical_seam(energy)) == 10


@pytest.mark.parametrize("seed", range(10))
def test_lookahead_depth_1_is_plain_greedy(seed):
    energy = np.random.default_rng(seed).random((15, 10))
    np.testing.assert_array_equal(find_vertical_seam_greedy_lookahead(energy, 1),
                                  find_vertical_seam_greedy(energy))


@pytest.mark.parametrize("seed", range(10))
def test_full_lookahead_is_optimal(seed):
    # With depth = m the lookahead table is the (bottom-up) DP table.
    energy = np.random.default_rng(seed).random((12, 9))
    seam = find_vertical_seam_greedy_lookahead(energy, depth=12)
    assert vertical_cost(energy, seam) == pytest.approx(
        vertical_cost(energy, find_vertical_seam(energy)))


def test_lookahead_sees_the_trap():
    # depth 2 sees row 1, so the trap's 9s are visible from row 0.
    assert vertical_cost(TRAP, find_vertical_seam_greedy_lookahead(TRAP, 2)) == 4


def test_lookahead_costs_hand_worked():
    #   e = [[1, 2, 2], [9, 9, 1], [9, 9, 1]], depth 2:
    #   row 0: 1+min(9,9)=10, 2+min(9,9,1)=3, 2+min(9,1)=3 ; rows 1: e+min(below) ; row 2: e
    L = lookahead_costs(TRAP, 2)
    np.testing.assert_array_equal(L, [[10, 3, 3], [18, 10, 2], [9, 9, 1]])


def test_lookahead_rejects_bad_depth():
    with pytest.raises(ValueError):
        find_vertical_seam_greedy_lookahead(TRAP, 0)
