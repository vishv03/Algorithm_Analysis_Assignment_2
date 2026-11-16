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

## Sample images

- `samples/scene.png`: synthetic test scene generated for this project.
- `samples/broadway_tower.jpg`: Broadway Tower, Worcestershire. Photo by
  [Newton2](https://en.wikipedia.org/wiki/User:Newton2), cropped by Yummifruitbat,
  from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Broadway_tower_edit.jpg),
  licensed [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/). Downscaled to 960 x 651.

```
.venv\Scripts\python cli.py samples/broadway_tower.jpg out/broadway_w640.jpg --width 640
```
