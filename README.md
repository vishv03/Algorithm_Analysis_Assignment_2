# Seam Carving with Dynamic Programming

[![CI/CD](https://github.com/vishv03/Algorithm_Analysis_Assignment_2/actions/workflows/ci.yml/badge.svg)](https://github.com/vishv03/Algorithm_Analysis_Assignment_2/actions/workflows/ci.yml)

Content-aware image resizing (Avidan & Shamir, 2007), Assignment 2 Question 1.
Removes low-energy vertical and horizontal seams, found with dynamic programming,
until the image reaches the requested size. Two alternatives to DP are also
implemented and compared: a greedy seam finder (Question 3) and a graph cut
(min-cut / max-flow) seam finder.

## Setup

On Windows, use backslashes in the path (`.venv\Scripts\python`); cmd does not
understand `.venv/Scripts/python`. On macOS / Linux use `.venv/bin/python`.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows (cmd or PowerShell)
# .venv/bin/python -m pip install -r requirements.txt     # macOS / Linux
```

The graph cut seam finder needs `scipy` for max-flow (in `requirements.txt`).

## Tests

```
.venv\Scripts\python -m pytest -q
```

## Usage

```
.venv\Scripts\python cli.py INPUT OUTPUT [--width W] [--height H]
```

Examples, using the included synthetic sample (480 x 300):

```
.venv\Scripts\python cli.py samples/scene.png out/scene_w300.png --width 300
.venv\Scripts\python cli.py samples/scene.png out/scene_300x220.png --width 300 --height 220
```

An omitted dimension stays unchanged. The image can only be shrunk.
Add `--energy sobel` to use the Sobel filter instead of the default dual-gradient
energy (both `cli.py` and `compare.py` accept it).
Pillow is used only to read and write files; the energy map, the DP seam
search and the seam removal are all implemented in `seamcarve/`.

## Comparison with scaling and cropping

```
.venv\Scripts\python compare.py samples/broadway_tower.jpg out/broadway_compare.jpg --width 640
```

Writes one image with four labelled panels: original, plain scaling, centred
crop, and seam carving (with its running time). Scaling and cropping use Pillow
only as baselines; the seam-carved panel uses `seamcarve.resize()`.

## How it works

| Step | Module | What it does |
|------|--------|--------------|
| 1a Energy map | `seamcarve/energy.py` | Dual-gradient energy (Δx² + Δy²) or Sobel energy (Gx² + Gy², 3x3 kernels), summed over R, G, B, with edge-replicated borders |
| 1b Minimum seam | `seamcarve/dp.py` | DP: `M[i][j] = e[i][j] + min(M[i-1][j-1], M[i-1][j], M[i-1][j+1])`, then backtrack. Horizontal seams reuse the same code on the transposed energy map |
| 1c Remove and repeat | `seamcarve/carve.py` | Delete the seam with a boolean mask, recompute the energy, repeat. Vertical and horizontal seams alternate while both dimensions shrink |

Per seam: O(m·n) time and space. Removing k seams: O(k·m·n) time.

## Improvements (`--fast`)

`cli.py ... --fast` turns on two optimisations. Both give **exactly** the same
output as the default (checked by `tests/test_improvements.py`):

- **Local energy update** (`resize(..., incremental=True)`): after removing a seam,
  only the pixels next to it can change energy. The seam is removed from the energy
  map too, and only a few pixels per row are recomputed: O(m) instead of O(m·n) for
  the energy step.
- **Low-memory DP** (`find_vertical_seam_low_memory`): row i of the DP table only needs
  row i-1, so the float64 table M is not stored. Only the int8 back-pointers are kept:
  about 1 byte per pixel instead of 9, with the same seam. The minimum cost alone needs
  just O(n) memory (`min_vertical_seam_cost`).

## Sample images

- `samples/scene.png`: synthetic test scene generated for this project.
- `samples/broadway_tower.jpg`: Broadway Tower, Worcestershire. Photo by
  [Newton2](https://en.wikipedia.org/wiki/User:Newton2), cropped by Yummifruitbat,
  from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Broadway_tower_edit.jpg),
  licensed [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/). Downscaled to 960 x 651.

```
.venv\Scripts\python cli.py samples/broadway_tower.jpg out/broadway_w640.jpg --width 640
```

## Benchmark

```
.venv\Scripts\python benchmark.py
```

Scales the sample photo to several sizes, then measures the energy map, one DP
seam search (full table and low-memory), and the time per seam for the default and
`--fast` modes, plus the DP's peak memory. Writes `benchmark.csv`, `benchmark.md`
(a table for the report) and two charts, `time_per_seam.png` and `dp_memory.png`,
to `out/benchmark/`. Use `--repeats`, `--seams` and `--scales` to change the runs.
Requires `matplotlib` (in `requirements.txt`).

## Greedy algorithm (Question 3)

`seamcarve/greedy.py` builds the seam one row at a time: it starts at the cheapest
pixel in row 0, then in each next row takes the cheapest of the (up to) 3 pixels
below the current one. It never looks ahead, so it is fast (O(m + n) per seam, no
table) but not optimal. The DP seam is the minimum over all seams, so a greedy seam
can never cost less, and the cheapest pixel now can lead into an expensive region
(see the `TRAP` example in `tests/test_greedy.py`: greedy 19 vs DP 4).

Use it from the command line with `--finder greedy` (works with `--width`,
`--height`, `--energy` and `--fast`):

```
.venv\Scripts\python cli.py samples/broadway_tower.jpg out/broadway_greedy.jpg --width 672 --finder greedy
```

### Alternative greedy designs

Two variants test whether greedy can be fixed cheaply (`--finder greedy-multistart`
or `--finder greedy-lookahead`):

- **Multi-start** (`find_vertical_seam_greedy_multistart`): runs greedy from every
  top-row pixel (all at once with NumPy) and keeps the cheapest. Fixes a bad start,
  but costs O(m·n) and is still not optimal (see `test_multistart_is_still_not_optimal`).
- **Lookahead** (`find_vertical_seam_greedy_lookahead`, depth 3): scores each step by
  the cheapest 3-row path ahead, from a truncated bottom-up DP table (`lookahead_costs`).
  Costs O(depth·m·n); depth 1 is plain greedy and depth m is exactly DP.

Compare all of them with DP (seams, visual quality, seam energy, running time):

```
.venv\Scripts\python greedy_compare.py
```

This writes `seams_first.png` (first seam of each: DP red, greedy blue),
`seams_all.png` (first 40 seams of every method), `result.png` (original and every method),
`seam_cost.png`, `timing.png` and `greedy_vs_dp.md` (tables for the report) to
`out/greedy/`. Options: `--width`, `--seams-shown`, `--energy`, `--methods`, `--scales`, `--repeats`.

Check the conclusions on every sample image and both energy functions:

```
.venv\Scripts\python greedy_experiments.py
```

Writes `experiments.csv`, `experiments.md` and one result panel per
(image, energy) pair to `out/greedy/experiments/`.

Result on the Broadway Tower photo (960x651 -> 672x651): greedy finds each seam
about 20x faster, but the whole resize is only about 1.6x faster, because the energy
map and the seam removal (both O(m·n)) still run for every seam. Greedy seams cost
about 3x as much energy as DP seams on average (up to 20x), and it visibly
squeezes the tower, while DP takes its seams from the sky and grass.

## Graph cut (min-cut / max-flow)

`seamcarve/graphcut.py` reformulates seam finding as a minimum s–t cut. A vertical
seam splits the image into a left part (source side S) and a right part (sink
side T), so the cheapest cut is the cheapest seam. One node per pixel, plus a
source s and a sink t:

| Edge | Capacity | Purpose |
|------|----------|---------|
| s → (i, 0) | ∞ | the first column is always on the source side |
| (i, j) → (i, j+1) | e[i][j] | cutting it removes pixel (i, j), the only edges that cost anything |
| (i, n-1) → t | e[i][n-1] | lets the last column be removed too |
| (i, j+1) → (i, j) | ∞ | S is a prefix of each row, so the cut crosses each row exactly once |
| (i, j) → (i-1, j-1), (i+1, j-1) | ∞ | seam columns in neighbouring rows differ by at most 1 (connected seam) |

Max-flow is run with scipy's Dinic (default) or Edmonds–Karp, or with our own
Edmonds–Karp (`edmonds_karp()`, pure Python, used in the tests). A BFS from s in
the residual graph gives S, and the seam pixel in each row is the rightmost pixel
of that row in S. By the max-flow min-cut theorem the flow value equals the seam
energy, which the code asserts. Horizontal seams use the transposed energy map,
as with DP.

scipy's max-flow needs int32 capacities. Dual-gradient and Sobel energies of a
uint8 image are already integers and are used exactly when they fit; otherwise
they are scaled and rounded. ∞ is `m * max(e) + 1`, more than any straight seam
costs, so it is never cut.

`find_vertical_seam_graphcut` has the same signature as the DP finder and is
registered as `"graphcut"` in `SEAM_FINDERS`, so it plugs straight into
`resize(finder=...)`:

```
.venv\Scripts\python cli.py samples/broadway_tower.jpg out/broadway_graphcut.jpg --width 950 --finder graphcut
```

Graph cut is slow (see below): on the full 960 x 651 photo each seam can take 10 s or more,
so keep the number of seams small (10 here, a few minutes in total). `--fast` only changes the
energy update for graph cut; the low-memory DP applies to `--finder dp` only.

### Forward energy

`forward_energy` in `seamcarve/energy.py` is the cost suggested for the graph cut,
`|I(i, j+1) - I(i, j-1)|` summed over R, G, B. It measures the new edge created when
pixel (i, j) is removed, instead of the energy removed. It is **not** in
`ENERGY_FUNCTIONS`: it only looks left and right, so the energy of the transposed
image is not the transpose of the energy, which the transpose trick for horizontal
seams relies on. It is available as `--energy forward` in `graphcut_compare.py`, which
always computes it on the image being carved (transposed for horizontal seams).

### Comparison with DP

```
.venv\Scripts\python graphcut_compare.py
```

Shrinks the photo so its longer side is 300 px (`--max-side`, 0 = full size), removes
40 vertical and 40 horizontal seams (`--seams`) with DP and with graph cut, and times
one seam search at several sizes with DP, graph cut (Dinic) and graph cut
(Edmonds–Karp). Writes `seams_vertical.png`, `seams_horizontal.png` (DP red, graph cut
green), `result.png`, `timing.png` and `graphcut_vs_dp.md` (tables for the report) to
`out/graphcut/`. Options: `--energy`, `--scales`, `--repeats`, `--no-charts`.

Result on the Broadway Tower photo (300 x 203, dual-gradient, 40 seams each way):
graph cut finds exactly the same seams as DP (40/40 in both directions, identical
output), because both solve the same problem exactly. It is about 150–400x slower:
DP takes about 2–4 ms per seam, graph cut 570–810 ms (Dinic), and Edmonds–Karp is
about 10x slower again (3.3 s per seam at 225 x 152). DP is one O(m·n) pass; graph
cut runs a general max-flow on a graph with about 5·m·n edges. The advantage of graph
cut is generality, not speed: the same construction extends to seam surfaces in
video and to constraints DP cannot express.

Tests: `tests/test_graphcut.py` checks that the graph cut seam always has the same
energy as the DP seam (random maps, all energy functions, all three solvers), that
the first and last columns can be removed, the edge count, and `resize()` integration.

## Bonus

### (a) Step-by-step visualisation, with every seam

```
.venv\Scripts\python visualize.py samples/broadway_tower.jpg out/visualize --width 220 --height 160
```

Writes to `out/visualize/`:

| File | Content |
|------|---------|
| `steps.gif` | one frame per step (about 60 frames; `--every N` to choose), three panels: the current image with the seam about to be removed, the energy map of the current image with the same seam, and the original image with **all** seams removed so far |
| `summary.png` | original, every removed seam drawn on the original, result |
| `seams.csv` | step, direction, seam energy, size after each step (for charts in the report) |
| `result.png` | the carved image |
| `frames/step_0001.png` ... | every step as a PNG (`--frames`) |

Vertical seams are drawn red → yellow (first → last), horizontal seams blue → cyan.
The input is scaled so its longer side is 320 px (`--max-side`, 0 = full size), because
a GIF of a large photo gets big quickly. `--energy`, `--finder` and `--fast` work as in `cli.py`.

How the seams are drawn on the original: each seam is found in the *current*, already
narrower image, so its columns are not original positions. `seamcarve/visualize.py`
carries an index map (original position of every remaining pixel) through the carving
and removes each seam from it as well; looking the seam up in the index map before
removing it gives its original pixels. One extra O(m·n) copy per seam.

`resize()` and the visualisation share the same loop: `carve_steps()` in
`seamcarve/carve.py` yields one `SeamStep` (image, energy map, seam, result) per seam,
and `resize()` just keeps the last result, so the visualised seams are exactly the ones
`cli.py` removes.

### (b) Interactive resizing

```
.venv\Scripts\python interactive.py samples/broadway_tower.jpg
```

Opens `http://localhost:8000` in the browser. Drag the width / height sliders or the
handle at the corner of the image and the seam-carved result updates live (a few ms per
update). The page also shows the removed seams on the original, a scaling or cropping
baseline at the same size, an **Animate** button that replays the carving, the energy
function choice, **Open image…** (or drag and drop a file), and **Download full size**,
which carves the full-resolution image to the chosen size. The server uses only the
Python standard library (`http.server`); the page is `webapp/index.html`.

Why it is instant: carving each new size from scratch would cost O(k·m·n). Instead, as in
Avidan & Shamir's *multi-size images*, the server carves vertical seams down to the
minimum width once and records a **removal-order map**: the step at which every pixel is
removed. Width w is then just "keep the pixels removed at step n − w or later", one
O(m·n) mask that the browser applies itself. For the height, the server sends a
horizontal removal-order map of the width-w image (computed once per width, cached), and
the browser masks it the same way. The result for (w, h) is exactly
`resize(resize(img, m, w), h, w)`: vertical seams first, then horizontal (the tests check
this, and so does the download, which uses the same order at full size).

The live preview works on a copy scaled to 480 px (`--preview-size`); sizes down to 25 %
are offered (`--min-scale`). Precomputing takes about 1.5 s for the sample photo.

### (c) CI/CD

`.github/workflows/ci.yml` (GitHub Actions):

| Job | When | What |
|-----|------|------|
| Lint | every push / PR | `ruff`: syntax errors, undefined names, unused imports and variables |
| Tests | every push / PR | full `pytest` suite on Ubuntu (Python 3.10, 3.12, 3.14), Windows and macOS; JUnit report uploaded |
| Demo | after lint and tests pass | runs `cli.py`, `compare.py`, `visualize.py`, `greedy_compare.py`, `graphcut_compare.py` and `benchmark.py` on the samples; uploads everything in `out/` as the `demo-outputs` artifact and writes a summary (seam statistics, benchmark table) on the run page |
| Release | on a `v*` tag | creates a GitHub Release with the demo outputs, the step-by-step GIF and a source zip attached |

Make a release with `git tag v1.0 && git push origin v1.0`.

Tests for the bonus features: `tests/test_visualize.py` (removal-order maps reproduce
`resize()` for every size, tracked seams are exactly the removed pixels, the
visualisation outputs, and the interactive server's API end to end).
