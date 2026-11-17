"""Graph cut (min s-t cut / max-flow): another alternative to DP for finding a seam.

Idea
    A vertical seam splits the image into a LEFT part and a RIGHT part. So
    finding a seam = cutting a graph of pixels into a source side S (left) and
    a sink side T (right). If cutting costs exactly the energy of the removed
    pixels, the CHEAPEST cut is the CHEAPEST seam. Max-flow finds the cheapest
    cut (max-flow min-cut theorem).

Graph for a vertical seam (m rows, n columns)
    One node per pixel, node id of (i, j) = i*n + j, plus source s and sink t.

        edge                            capacity   why
        s -> (i, 0)                     INF        column 0 is always on the left (S)
        (i, j) -> (i, j+1)              e[i, j]    cutting it removes pixel (i, j)
        (i, n-1) -> t                   e[i, n-1]  lets the LAST column be removed too
        (i, j+1) -> (i, j)              INF        S is a prefix of each row:
                                                   the cut crosses each row ONCE
        (i, j) -> (i-1, j-1)            INF        seam columns in neighbouring rows
        (i, j) -> (i+1, j-1)            INF        differ by at most 1: CONNECTED seam

    An INF edge can never be cut (the cut would cost "infinity"), so INF edges
    are hard constraints. Only the e[i, j] edges are ever paid for.

Why the constraints give exactly the DP seams
    * Backward INF edge: if (i, j+1) is in S, then (i, j) is in S. So S is
      columns 0..c_i of row i (a prefix), and exactly one cost edge is cut in
      row i: (i, c_i) -> (i, c_i + 1). The removed pixel is (i, c_i).
    * Diagonal INF edges: (i, c_i) in S forces (i±1, c_i - 1) in S, so
      c_{i±1} >= c_i - 1. Applied both ways, |c_i - c_{i+1}| <= 1.
    So every finite cut is a valid seam, every seam is a finite cut, and
    cut capacity = sum of e over the seam. Min cut = minimum seam, the same
    problem the DP solves (equal cost; the path can differ only on ties).

Reading the seam from the max-flow
    After max-flow, take the residual graph (capacity - flow) and BFS from s
    along edges that still have capacity left. The nodes reached are S.
    In each row, seam[i] = rightmost pixel of the row in S = (size of S in row i) - 1.

Integer capacities
    scipy's max-flow needs int32 capacities. Dual-gradient and Sobel energies
    of a uint8 image are already integers, so they are used exactly whenever
    they fit. Otherwise the energy is scaled down and rounded (see
    _integer_costs), and the seam is optimal for that rounded energy.
    INF only has to beat the cost of every possible cut; any straight seam
    costs at most m * max(e), so INF = m * max(e) + 1 is enough.

Complexity (m rows, n columns; V = m·n + 2 nodes, E ≈ 5·m·n edges)
    Building the graph: O(m·n).
    Max-flow: Edmonds-Karp O(V·E²), Dinic O(V²·E) in the worst case. Much
    slower than the O(m·n) DP in practice (see graphcut_compare.py).
    Space: O(m·n) for the graph.
    The advantage is generality, not speed: the same construction works for
    2-D seam surfaces in video and for extra constraints that DP cannot handle.

Horizontal seams
    Same trick as dp.py: a horizontal seam of `energy` is a vertical seam of
    `energy.T`.
"""

from collections import deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

INT32_MAX = 2**31 - 1
SOLVERS = ("dinic", "edmonds_karp", "own")   # "own" = our Edmonds-Karp below


# ---------------------------------------------------------------- costs

def _integer_costs(energy: np.ndarray) -> np.ndarray:
    """Energy -> non-negative int64 costs such that m * max(cost) + 1 fits in int32."""
    m = energy.shape[0]
    limit = (INT32_MAX - 1) // m                    # largest allowed cost per pixel
    e = np.maximum(np.asarray(energy, dtype=np.float64), 0.0)
    top = float(e.max())
    if top <= limit and np.all(e == np.round(e)):
        return e.astype(np.int64)                   # exact: already integers that fit
    return np.round(e * (limit / max(top, 1e-12))).astype(np.int64)


# ---------------------------------------------------------------- graph

def build_graph(cost: np.ndarray) -> tuple[sp.csr_matrix, int, int]:
    """Build the seam graph described in the module docstring.

    Args:
        cost: int64 array (m, n), cost of removing each pixel.

    Returns:
        (capacity, s, t): capacity is an (m·n+2) x (m·n+2) int32 CSR matrix,
        capacity[u, v] = capacity of edge u -> v. s = m·n, t = m·n + 1.
    """
    m, n = cost.shape
    s, t = m * n, m * n + 1
    inf = m * int(cost.max()) + 1                    # beats every possible cut
    node = np.arange(m * n).reshape(m, n)            # node[i, j] = id of pixel (i, j)

    tails, heads, caps = [], [], []

    def add(u, v, c):
        """Add edges u -> v with capacity c (arrays of the same shape, or scalar c)."""
        u, v = np.asarray(u), np.asarray(v)
        tails.append(u.ravel())
        heads.append(v.ravel())
        caps.append(np.broadcast_to(np.asarray(c, dtype=np.int64), u.shape).ravel())

    add(np.full(m, s), node[:, 0], inf)               # s -> first column
    add(node[:, :-1], node[:, 1:], cost[:, :-1])      # (i, j) -> (i, j+1), cost e[i, j]
    add(node[:, -1], np.full(m, t), cost[:, -1])      # last column -> t, cost e[i, n-1]
    add(node[:, 1:], node[:, :-1], inf)               # backward (i, j+1) -> (i, j)
    add(node[1:, 1:], node[:-1, :-1], inf)            # diagonal (i, j) -> (i-1, j-1)
    add(node[:-1, 1:], node[1:, :-1], inf)            # diagonal (i, j) -> (i+1, j-1)

    u, v, c = np.concatenate(tails), np.concatenate(heads), np.concatenate(caps)
    # Also store every reverse pair (v, u) with capacity 0, so that
    # "capacity - flow" gives the residual capacity in BOTH directions.
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.concatenate([c, np.zeros_like(c)])
    capacity = sp.csr_matrix((data, (rows, cols)), shape=(m * n + 2, m * n + 2))
    capacity.sum_duplicates()          # no real duplicates; merges the 0 placeholders
    return capacity.astype(np.int32), s, t


# ---------------------------------------------------------------- max-flow

def edmonds_karp(capacity: sp.csr_matrix, s: int, t: int) -> tuple[int, sp.csr_matrix]:
    """Our own Edmonds-Karp (Ford-Fulkerson with BFS paths), for small graphs and tests.

    Repeat: BFS from s in the residual graph for the SHORTEST augmenting path
    to t, push its bottleneck capacity along it (subtract on each edge, add on
    the reverse edge). Stop when t is unreachable.

    Returns (max-flow value, residual capacities as a CSR matrix).
    Pure Python, so only use it on small images; resize() uses scipy.
    """
    capacity = capacity.tocsr()
    start, neighbour = capacity.indptr, capacity.indices
    residual = capacity.data.astype(np.int64).copy()
    num_nodes = capacity.shape[0]

    # reverse[k] = index of edge v -> u, for the CSR entry k of edge u -> v.
    position = {}
    for u in range(num_nodes):
        for k in range(start[u], start[u + 1]):
            position[(u, neighbour[k])] = k
    reverse = np.empty(len(residual), dtype=np.int64)
    for (u, v), k in position.items():
        reverse[k] = position[(v, u)]

    flow = 0
    while True:
        came_by = np.full(num_nodes, -1)            # edge used to reach each node
        came_by[s] = -2
        queue = deque([s])
        while queue and came_by[t] == -1:           # BFS = shortest path in edges
            u = queue.popleft()
            for k in range(start[u], start[u + 1]):
                v = neighbour[k]
                if residual[k] > 0 and came_by[v] == -1:
                    came_by[v] = k
                    queue.append(v)
        if came_by[t] == -1:
            break                                   # no augmenting path: flow is maximum
        path, v = [], t
        while v != s:                               # walk back from t to s
            k = came_by[v]
            path.append(k)
            v = neighbour[reverse[k]]               # tail of edge k
        bottleneck = min(residual[k] for k in path)
        for k in path:
            residual[k] -= bottleneck
            residual[reverse[k]] += bottleneck
        flow += bottleneck

    return flow, sp.csr_matrix((residual, neighbour, start), shape=capacity.shape)


def max_flow_residual(capacity, s, t, solver="dinic") -> tuple[int, sp.csr_matrix]:
    """Max-flow with the chosen solver. Returns (flow value, residual capacities)."""
    if solver == "own":
        return edmonds_karp(capacity, s, t)
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}, choose from {SOLVERS}")
    result = maximum_flow(capacity, s, t, method=solver)
    # scipy's flow is antisymmetric (flow[v, u] = -flow[u, v]), so this gives
    # the residual capacity of every edge AND of every reverse edge.
    residual = capacity.astype(np.int64) - result.flow.astype(np.int64)
    return int(result.flow_value), residual.tocsr()


# ---------------------------------------------------------------- seam

def find_vertical_seam_graphcut(energy: np.ndarray, solver: str = "dinic") -> np.ndarray:
    """Return a minimum-energy vertical seam (one column index per row) via min cut.

    Same interface as dp.find_vertical_seam, so it plugs into resize().
    """
    if energy.ndim != 2:
        raise ValueError(f"energy map must be 2-D, got shape {energy.shape}")
    m, n = energy.shape
    if m == 0 or n == 0:
        raise ValueError("energy map must have at least one row and one column")

    cost = _integer_costs(energy)
    capacity, s, t = build_graph(cost)
    flow, residual = max_flow_residual(capacity, s, t, solver)

    # S = everything reachable from s using edges with residual capacity > 0.
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reached = breadth_first_order(residual, s, directed=True, return_predecessors=False)
    in_S = np.zeros(m * n + 2, dtype=bool)
    in_S[reached] = True
    S = in_S[: m * n].reshape(m, n)

    # S is a prefix of every row, so the rightmost S pixel is (count in S) - 1.
    seam = S.sum(axis=1).astype(np.intp) - 1

    # Sanity checks: a real min cut, a valid seam, and cut capacity = seam cost.
    assert not in_S[t], "sink reachable: the flow is not maximum"
    assert np.all(S[np.arange(m), seam]) and np.all(np.abs(np.diff(seam)) <= 1)
    assert flow == int(cost[np.arange(m), seam].sum()), "cut capacity != seam cost"
    return seam


def find_horizontal_seam_graphcut(energy: np.ndarray, solver: str = "dinic") -> np.ndarray:
    """Return a minimum-energy horizontal seam (one row index per column)."""
    return find_vertical_seam_graphcut(energy.T, solver)
