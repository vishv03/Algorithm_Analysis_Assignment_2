"""Tests for the graph cut seam finder: it must always find a seam as cheap as DP's."""

import numpy as np
import pytest

from seamcarve.carve import resize
from seamcarve.dp import find_vertical_seam
from seamcarve.energy import dual_gradient_energy, forward_energy, sobel_energy
from seamcarve.graphcut import (
    build_graph,
    find_horizontal_seam_graphcut,
    find_vertical_seam_graphcut,
)


def seam_cost(energy, seam):
    return energy[np.arange(len(seam)), seam].sum()


def test_greedy_trap_example():
    # The example from greedy.py: greedy pays 19, the optimum is 4.
    e = np.array([[1, 2, 2], [9, 9, 1], [9, 9, 1]], dtype=float)
    # Two seams cost 4 ([1, 2, 2] and [2, 2, 2]), so check the cost, not the path.
    assert seam_cost(e, find_vertical_seam_graphcut(e)) == 4
    assert seam_cost(e, find_vertical_seam_graphcut(e, solver="own")) == 4


def test_last_and_first_column_can_be_the_seam():
    e = np.full((4, 5), 9.0)
    e[:, -1] = 0                      # needs the (i, n-1) -> t cost edge
    assert list(find_vertical_seam_graphcut(e)) == [4, 4, 4, 4]
    e = np.full((4, 5), 9.0)
    e[:, 0] = 0
    assert list(find_vertical_seam_graphcut(e)) == [0, 0, 0, 0]


@pytest.mark.parametrize("solver", ["dinic", "edmonds_karp", "own"])
def test_same_cost_as_dp_on_random_maps(solver):
    rng = np.random.default_rng(0)
    for _ in range(40):
        m, n = rng.integers(1, 8), rng.integers(1, 8)
        e = rng.integers(0, 100, (m, n)).astype(float)
        seam = find_vertical_seam_graphcut(e, solver)
        assert seam.shape == (m,)
        assert np.all(np.abs(np.diff(seam)) <= 1)
        assert seam_cost(e, seam) == seam_cost(e, find_vertical_seam(e))


@pytest.mark.parametrize("energy_fn", [dual_gradient_energy, sobel_energy, forward_energy])
def test_same_cost_as_dp_on_image(energy_fn):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    e = energy_fn(img)
    assert seam_cost(e, find_vertical_seam_graphcut(e)) == seam_cost(e, find_vertical_seam(e))
    assert seam_cost(e.T, find_horizontal_seam_graphcut(e)) == seam_cost(e.T, find_vertical_seam(e.T))


def test_edge_count():
    m, n = 4, 6
    capacity, s, t = build_graph(np.ones((m, n), dtype=np.int64))
    # real edges are the non-zero entries (the zero ones are reverse placeholders)
    real = int((capacity.data > 0).sum())
    assert real == m + m * (n - 1) + m + m * (n - 1) + 2 * (m - 1) * (n - 1)


def test_plugs_into_resize():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (20, 25, 3), dtype=np.uint8)
    out = resize(img, 17, 20, finder=find_vertical_seam_graphcut)
    assert out.shape == (17, 20, 3)
    assert np.array_equal(out, resize(img, 17, 20))  # same seams as DP on this image
