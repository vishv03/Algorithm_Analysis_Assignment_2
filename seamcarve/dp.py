"""Step 1b: find the minimum-energy seam with dynamic programming.

A *vertical seam* is a path from the top row to the bottom row that takes
exactly one pixel per row, where the pixels in consecutive rows are in the
same or an adjacent column (|seam[i] - seam[i-1]| <= 1). Its cost is the sum
of the energies of its pixels.

There are exponentially many seams (at least 2^(m-1) when n > 1), so trying
them all is hopeless. Dynamic programming finds the cheapest one in O(m·n).

Subproblem
    M[i][j] = the minimum cost of any seam that starts anywhere in row 0
              and ends at pixel (i, j).

Recurrence
    M[0][j] = e[0][j]
    M[i][j] = e[i][j] + min( M[i-1][j-1], M[i-1][j], M[i-1][j+1] )
    (a neighbour outside the image counts as +infinity, so it is never chosen)

Why it is correct (optimal substructure)
    The pixel above (i, j) on any seam must be (i-1, j-1), (i-1, j) or
    (i-1, j+1). If the best seam ending at (i, j) came from one of them, the
    part of it above row i must itself be the best seam ending at that pixel,
    otherwise we could swap in a cheaper prefix and get a cheaper seam.
    Many seams share the same prefixes (overlapping subproblems), and the
    table computes each prefix only once.

Answer
    The best seam ends at the column with the smallest value in the last row,
    min_j M[m-1][j]. To recover the actual path we store, for every cell, which
    of the three cells above was the minimum (a "back-pointer" -1, 0 or +1),
    then walk the back-pointers from the bottom row up to the top.

Horizontal seams
    A horizontal seam (left to right, one pixel per column) in an image is
    exactly a vertical seam in the transposed image. So we transpose the
    energy map and reuse the same code: no second implementation needed.

Complexity (m rows, n columns)
    Time:  O(m·n) - each cell looks at 3 cells above; backtracking is O(m).
    Space: O(m·n) - the M table and the back-pointer table.
"""

import numpy as np


def cumulative_energy_vertical(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill the DP tables for vertical seams.

    Args:
        energy: float array of shape (m, n), the energy map.

    Returns:
        M:    float64 (m, n), M[i, j] = cheapest cost of a seam ending at (i, j).
        back: int8 (m, n),   back[i, j] in {-1, 0, +1} is the column offset of
              the best pixel in row i-1, i.e. the seam continues upward to
              column j + back[i, j]. Row 0 has no row above, so it stays 0.
    """
    _check_energy(energy)
    m, n = energy.shape

    M = np.empty((m, n), dtype=np.float64)
    back = np.zeros((m, n), dtype=np.int8)
    M[0] = energy[0]                      # base case: a seam of length 1

    for i in range(1, m):                 # rows must be done in order (row i needs row i-1)
        M[i], back[i] = _next_row(M[i - 1], energy[i])

    return M, back


def _next_row(prev: np.ndarray, energy_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One step of the recurrence: compute row i of M from row i-1.

    Args:
        prev:       M[i-1], length n.
        energy_row: e[i],   length n.

    Returns:
        (M[i], back[i]) - the new DP row and its back-pointers in {-1, 0, +1}.
    """
    n = prev.shape[0]

    # For every column j at once, build the three candidates from the row above:
    #   up_left[j]  = M[i-1, j-1]  (inf for j = 0, no pixel to the left)
    #   up[j]       = M[i-1, j]
    #   up_right[j] = M[i-1, j+1]  (inf for j = n-1, no pixel to the right)
    up_left = np.empty(n)
    up_left[0] = np.inf
    up_left[1:] = prev[:-1]

    up_right = np.empty(n)
    up_right[-1] = np.inf
    up_right[:-1] = prev[1:]

    candidates = np.stack([up_left, prev, up_right])   # shape (3, n)

    # Index 0, 1 or 2 of the smallest candidate in each column.
    # On a tie, argmin returns the first one, so ties prefer up-left,
    # then straight up. Any choice is fine: all give the same cost.
    best = np.argmin(candidates, axis=0)

    new_row = energy_row + candidates[best, np.arange(n)]   # the recurrence
    return new_row, (best - 1).astype(np.int8)              # 0,1,2 -> -1,0,+1


def backtrack_vertical(M: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Recover the minimum seam from the filled DP tables.

    Starts at the cheapest cell in the bottom row and follows back-pointers
    up to row 0. Only the LAST row of M is actually needed here.

    Returns:
        int array of length m: seam[i] is the column of the seam in row i.
    """
    return _backtrack(M[-1], back)


def _backtrack(last_row: np.ndarray, back: np.ndarray) -> np.ndarray:
    m = back.shape[0]
    seam = np.empty(m, dtype=np.intp)
    j = int(np.argmin(last_row))          # end of the cheapest seam
    seam[-1] = j
    for i in range(m - 1, 0, -1):         # rows m-1 down to 1
        j += int(back[i, j])              # step to the chosen pixel in row i-1
        seam[i - 1] = j
    return seam


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Return the minimum-energy vertical seam (one column index per row)."""
    M, back = cumulative_energy_vertical(energy)
    return backtrack_vertical(M, back)


# ---------------------------------------------------------------------------
# Improvement: do we need the entire DP table?
#
# Row i of M depends only on row i-1. So:
#   * The minimum seam COST needs just two rows: O(n) extra space.
#   * The seam PATH still needs to know, for every cell, where it came from.
#     We keep only the int8 back-pointer table (1 byte per pixel) and drop the
#     float64 M table (8 bytes per pixel): about 9x less memory, same O(m·n)
#     time and exactly the same seam (same tie-breaking).
#   * Going below O(m·n) space for the path is possible (Hirschberg-style
#     divide and conquer recomputes rows instead of storing pointers), but it
#     costs O(m·n·log m) time, so it is not worth it here.
# ---------------------------------------------------------------------------

def min_vertical_seam_cost(energy: np.ndarray) -> float:
    """Cost of the minimum vertical seam using only O(n) extra memory."""
    _check_energy(energy)
    row = energy[0].astype(np.float64)
    for i in range(1, energy.shape[0]):
        row, _ = _next_row(row, energy[i])      # keep only the latest row
    return float(row.min())


def find_vertical_seam_low_memory(energy: np.ndarray) -> np.ndarray:
    """Same seam as find_vertical_seam, without storing the full M table.

    Keeps one float64 row of M plus the int8 back-pointer table.
    Space: m·n bytes + O(n), versus 9·m·n bytes for the full-table version.
    """
    _check_energy(energy)
    m, n = energy.shape
    back = np.zeros((m, n), dtype=np.int8)
    row = energy[0].astype(np.float64)
    for i in range(1, m):
        row, back[i] = _next_row(row, energy[i])
    return _backtrack(row, back)


def find_horizontal_seam(energy: np.ndarray) -> np.ndarray:
    """Return the minimum-energy horizontal seam (one row index per column).

    A horizontal seam of `energy` is a vertical seam of `energy.T`
    (rows and columns swap roles), so we simply reuse the vertical version.
    """
    return find_vertical_seam(energy.T)


def _check_energy(energy: np.ndarray) -> None:
    if energy.ndim != 2:
        raise ValueError(f"energy map must be 2-D, got shape {energy.shape}")
    if energy.shape[0] == 0 or energy.shape[1] == 0:
        raise ValueError("energy map must have at least one row and one column")
