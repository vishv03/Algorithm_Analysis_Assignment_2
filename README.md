# Seam Carving with Dynamic Programming

Content-aware image resizing (Avidan & Shamir, 2007), Assignment 2 Question 1.
Removes low-energy vertical and horizontal seams, found with dynamic programming,
until the image reaches the requested size.

## Setup

On Windows, use backslashes in the path (`.venv\Scripts\python`); cmd does not
understand `.venv/Scripts/python`. On macOS / Linux use `.venv/bin/python`.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows (cmd or PowerShell)
# .venv/bin/python -m pip install -r requirements.txt     # macOS / Linux
```

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
