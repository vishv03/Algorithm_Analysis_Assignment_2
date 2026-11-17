"""Question 3: find a seam with a GREEDY algorithm (alternative to DP).

Idea
    Build the seam one row at a time, always taking the cheapest pixel that is
    allowed right now, without looking at the rows still to come:

        seam[0] = argmin_j e[0][j]                         (cheapest pixel in row 0)
        seam[i] = argmin over j in {seam[i-1]-1, seam[i-1], seam[i-1]+1}
                  (inside the image) of e[i][j]

    The result is always a valid seam (one pixel per row, consecutive pixels in
    the same or an adjacent column), so it plugs straight into resize().

Why it is NOT optimal
    Seam finding has optimal substructure (that is what DP uses), but it does
    NOT have the greedy-choice property: the pixel that is cheapest now can
    lead into a region where every continuation is expensive. Greedy commits
    to that pixel and can never undo it. DP instead keeps, for EVERY pixel,
    the best seam ending there, so it never gets trapped.

    Smallest example (see tests/test_greedy.py):

        e = [[1, 2, 2],         greedy: 1 -> 9 -> 9      = 19
             [9, 9, 1],         DP:     2 -> 1 -> 1      =  4
             [9, 9, 1]]

    The cheap pixel "1" in row 0 is a trap: once greedy takes it, column 2 is
    out of reach for row 1, and it pays 9 in every later row.

    What we can say for sure: cost(greedy seam) >= cost(DP seam), because the
    DP seam is the minimum over all seams and the greedy seam is one of them.
    There is no constant bound on the ratio (make the 9s larger).

Tie-breaking
    np.argmin returns the first minimum, so ties go to the left-most candidate
    (up-left before straight before right), the same order the DP uses.

Complexity (m rows, n columns), given the energy map
    Time:  O(n) for row 0, then O(1) per row (3 candidates) -> O(m + n).
    Space: O(m) for the seam itself, no table at all.
    Compare DP: O(m·n) time and O(m·n) space per seam.

    Note: inside resize() the energy map (O(m·n)) and the seam removal copy
    (O(m·n)) still run for every seam, so the WHOLE pipeline is still
    O(m·n) per seam with greedy. Greedy only removes the DP part.

Horizontal seams
    Same trick as the DP: a horizontal seam of `energy` is a vertical seam of
    `energy.T`, so we reuse the vertical version.
"""

import numpy as np


def find_vertical_seam_greedy(energy: np.ndarray) -> np.ndarray:
    """Return a vertical seam chosen greedily, row by row (one column per row).

    Args:
        energy: float array of shape (m, n), the energy map.

    Returns:
        int array of length m: seam[i] is the column of the seam in row i.
    """
    if energy.ndim != 2:
        raise ValueError(f"energy map must be 2-D, got shape {energy.shape}")
    m, n = energy.shape
    if m == 0 or n == 0:
        raise ValueError("energy map must have at least one row and one column")

    seam = np.empty(m, dtype=np.intp)
    j = int(np.argmin(energy[0]))         # start at the cheapest pixel of row 0
    seam[0] = j

    for i in range(1, m):
        # Allowed pixels in row i: columns j-1, j, j+1, clipped to the image.
        lo = max(j - 1, 0)
        hi = min(j + 1, n - 1)
        # Local decision: take the cheapest of (at most) 3 candidates.
        j = lo + int(np.argmin(energy[i, lo:hi + 1]))
        seam[i] = j

    return seam


def find_horizontal_seam_greedy(energy: np.ndarray) -> np.ndarray:
    """Return a greedy horizontal seam (one row index per column)."""
    return find_vertical_seam_greedy(energy.T)


# ---------------------------------------------------------------------------
# Alternative greedy designs (considered and measured in greedy_compare.py).
#
# Plain greedy fails for two reasons: (1) it commits to ONE start pixel, and
# (2) it looks only ONE row ahead. Each variant below fixes one of them, and
# each one turns out to cost as much as DP, O(m·n), while still not being
# optimal. That is the main argument for DP: once greedy is made robust, it is
# no longer cheaper.
# ---------------------------------------------------------------------------

def _check(energy: np.ndarray) -> None:
    if energy.ndim != 2:
        raise ValueError(f"energy map must be 2-D, got shape {energy.shape}")
    if energy.shape[0] == 0 or energy.shape[1] == 0:
        raise ValueError("energy map must have at least one row and one column")


def find_vertical_seam_greedy_multistart(energy: np.ndarray) -> np.ndarray:
    """Variant 1 - multi-start greedy: run greedy from EVERY top-row pixel, keep the cheapest.

    Fixes the bad start (the trap example), because the start that leads to the
    cheap column is also tried. Still not optimal: every one of the n runs makes
    the same one-row-ahead choices, so a trap in the MIDDLE of the image still
    catches all of them.

    All n runs are advanced together, one row at a time, with NumPy: run s is
    at column pos[s], and its 3 candidates are pos[s]-1, pos[s], pos[s]+1
    (clipped). A clipped candidate repeats the centre one, and argmin picks
    the first minimum, so every run makes exactly the same choices as
    find_vertical_seam_greedy started from column s.

    Time:  O(m·n) - n runs of m steps.   Space: O(m·n) for the n paths.
    """
    _check(energy)
    m, n = energy.shape
    paths = np.empty((m, n), dtype=np.intp)     # paths[:, s] = run started at column s
    pos = np.arange(n)
    paths[0] = pos
    total = energy[0].astype(np.float64)

    offsets = np.array([-1, 0, 1])
    for i in range(1, m):
        cand = np.clip(pos[np.newaxis, :] + offsets[:, np.newaxis], 0, n - 1)   # (3, n)
        vals = energy[i][cand]                                                  # (3, n)
        best = np.argmin(vals, axis=0)                  # first minimum = left-most
        pos = cand[best, np.arange(n)]
        total += vals[best, np.arange(n)]
        paths[i] = pos

    return paths[:, int(np.argmin(total))].copy()


def lookahead_costs(energy: np.ndarray, depth: int) -> np.ndarray:
    """L[i, j] = cheapest cost of a downward path of `depth` rows starting at (i, j).

    Rows near the bottom have fewer than `depth` rows below them; their path
    simply stops at the last row. Computed bottom-up with the same recurrence
    as the DP, but truncated to `depth` steps:

        L_1 = e
        L_k[i][j] = e[i][j] + min(L_{k-1}[i+1][j-1], L_{k-1}[i+1][j], L_{k-1}[i+1][j+1])

    Time: O(depth·m·n). With depth = m this IS the (bottom-up) DP table, which
    is exactly why more lookahead turns greedy back into dynamic programming.
    """
    L = energy.astype(np.float64)
    for _ in range(depth - 1):
        below = L[1:]                                    # L_{k-1} of rows i+1
        left = np.empty_like(below)
        left[:, 0] = np.inf
        left[:, 1:] = below[:, :-1]
        right = np.empty_like(below)
        right[:, -1] = np.inf
        right[:, :-1] = below[:, 1:]
        best_below = np.minimum(np.minimum(left, below), right)
        new = energy.astype(np.float64)                  # bottom row: path just stops
        new[:-1] += best_below
        L = new
    return L


def find_vertical_seam_greedy_lookahead(energy: np.ndarray, depth: int = 3) -> np.ndarray:
    """Variant 2 - lookahead greedy: choose each step by the best `depth`-row path ahead.

    Same walk as plain greedy, but a candidate pixel is scored by L[i, c], the
    cheapest path of `depth` rows starting at it, instead of by its own energy.
    depth = 1 is plain greedy. It sees traps up to depth-1 rows ahead, but not
    beyond. depth = m would be DP (see lookahead_costs).

    Time:  O(depth·m·n) for the lookahead table + O(m + n) for the walk.
    Space: O(m·n) for the table.
    """
    _check(energy)
    if depth < 1:
        raise ValueError("depth must be at least 1")
    m, n = energy.shape
    L = lookahead_costs(energy, depth)

    seam = np.empty(m, dtype=np.intp)
    j = int(np.argmin(L[0]))
    seam[0] = j
    for i in range(1, m):
        lo, hi = max(j - 1, 0), min(j + 1, n - 1)
        j = lo + int(np.argmin(L[i, lo:hi + 1]))
        seam[i] = j
    return seam
