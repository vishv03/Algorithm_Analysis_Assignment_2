"""Question 3: compare greedy seam finding (and two greedy variants) with DP.

Usage:
    python greedy_compare.py [IMAGE] [--out out/greedy] [--width W] [--seams-shown 40]
                             [--energy dual-gradient|sobel] [--methods ...]
                             [--scales ...] [--repeats 3] [--no-charts]

Removes vertical seams down to --width (default: 70% of the input width) with
every method, using the same energy function and the same resize loop, so the
only difference is how the seam is chosen. (Horizontal seams use the same code
on the transposed image, so they behave the same way.)

Methods (seamcarve.carve.SEAM_FINDERS):
    dp                  optimal seam (Question 1)
    greedy              row-by-row local choice (Question 3)
    greedy-multistart   greedy from every top-row pixel, keep the cheapest
    greedy-lookahead    greedy that scores each step by the best 3-row path ahead

Writes to the output folder:
    seams_first.png    the very first seam of DP (red) and greedy (blue)
    seams_all.png      the first --seams-shown seams removed by each method
    result.png         original and the result of each method, with running times
    seam_cost.png      energy of every removed seam, per method
    timing.png         time to find one seam vs image size, per method
    greedy_vs_dp.md    all the numbers as Markdown tables (paste into the report)
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark import SURFACE, TEXT, TEXT_2, GRID, _line_chart, median_ms
from seamcarve.carve import SEAM_FINDERS, remove_vertical_seam, resize
from seamcarve.energy import ENERGY_FUNCTIONS

GAP = 10
LABEL_HEIGHT = 24
DP_COLOR = (230, 30, 30)        # red
GREEDY_COLOR = (30, 90, 230)    # blue
METHODS = ("dp", "greedy", "greedy-multistart", "greedy-lookahead")
NAMES = {
    "dp": "Dynamic programming",
    "greedy": "Greedy",
    "greedy-multistart": "Greedy, multi-start",
    "greedy-lookahead": "Greedy, 3-row lookahead",
}
# One fixed colour per method in every chart (validated categorical slots).
COLORS = {"dp": "#2a78d6", "greedy": "#eb6834", "greedy-multistart": "#1d9e75",
          "greedy-lookahead": "#9b59b6"}


# ---------------------------------------------------------------- carving with bookkeeping

def carve_tracked(img: np.ndarray, new_width: int, finder, energy_fn) -> dict:
    """Remove vertical seams down to new_width and record what happened.

    Same steps as resize() (full energy recomputation every time), plus:
      * every removed seam in ORIGINAL image coordinates, so it can be drawn,
      * the energy of every removed seam (sum of e over its pixels),
      * the time spent in the finder alone, and in total.

    To map a seam back to the original image we carry an index map `cols`
    of the same shape as the image: cols[i, j] = original column of the pixel
    now at (i, j). It is carved with exactly the same seams as the image.
    """
    m, n = img.shape[:2]
    rows = np.arange(m)
    cols = np.tile(np.arange(n), (m, 1))
    out = img
    seams, costs, finder_s = [], [], 0.0

    start = time.perf_counter()
    while out.shape[1] > new_width:
        energy = energy_fn(out)
        t = time.perf_counter()
        seam = finder(energy)
        finder_s += time.perf_counter() - t

        costs.append(float(energy[rows, seam].sum()))
        seams.append(cols[rows, seam].copy())
        out = remove_vertical_seam(out, seam)
        cols = remove_vertical_seam(cols, seam)
    total_s = time.perf_counter() - start

    return {"image": np.ascontiguousarray(out), "seams": seams, "costs": costs,
            "total_s": total_s, "finder_s": finder_s}


def run_methods(img: np.ndarray, new_width: int, energy_fn, methods=METHODS) -> dict:
    """carve_tracked() for every method; returns {method: run}."""
    return {key: carve_tracked(img, new_width, SEAM_FINDERS[key], energy_fn) for key in methods}


# ---------------------------------------------------------------- drawing

def draw_seams(img: np.ndarray, seams, color, thickness: int = 1) -> np.ndarray:
    """Paint vertical seams (original coordinates) onto a copy of img."""
    out = img.copy()
    m, n = img.shape[:2]
    rows = np.arange(m)
    for seam in seams:
        for d in range(-(thickness // 2), thickness - thickness // 2):
            out[rows, np.clip(seam + d, 0, n - 1)] = color
    return out


def panels(items, columns: int = 2) -> Image.Image:
    """Lay out labelled panels in a grid: items = [(label, PIL image), ...]."""
    cols = min(columns, len(items))
    grid = [items[i:i + cols] for i in range(0, len(items), cols)]
    # Each panel keeps its own width, so narrower (carved) images sit side by side.
    row_w = [sum(p.width for _, p in row) + GAP * (len(row) - 1) for row in grid]
    row_h = [max(p.height for _, p in row) + LABEL_HEIGHT for row in grid]
    canvas = Image.new("RGB", (max(row_w), sum(row_h) + GAP * (len(grid) - 1)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row, h in zip(grid, row_h):
        x = 0
        for label, p in row:
            draw.text((x + 4, y + 6), label, fill="black")
            canvas.paste(p, (x, y + LABEL_HEIGHT))
            x += p.width + GAP
        y += h + GAP
    return canvas


def seam_cost_chart(path: Path, costs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for key, y in costs.items():
        y = np.asarray(y) / 1e6           # millions, so no "1e6" offset label
        ax.plot(np.arange(1, len(y) + 1), y, color=COLORS[key], linewidth=1.4, label=NAMES[key])
    ax.set_title("Energy of each removed seam, per method",
                 loc="left", fontsize=12, color=TEXT, pad=28)
    ax.set_xlabel("Seam number (in removal order)", fontsize=9, color=TEXT_2)
    ax.set_ylabel("Seam energy, millions (sum of e over its pixels)", fontsize=9, color=TEXT_2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=len(next(iter(costs.values()))))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=len(costs), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- timing vs size

def timing_rows(source: Image.Image, scales, energy_fn, repeats: int, methods=METHODS,
                seams: int = 10):
    """For each size: median time of one seam search, and of one full seam removal."""
    rows = []
    for scale in sorted(scales):
        size = (max(seams + 2, round(source.width * scale)), max(3, round(source.height * scale)))
        img = np.asarray(source.resize(size, Image.Resampling.LANCZOS))
        h, w = img.shape[:2]
        energy = energy_fn(img)
        row = {"width": w, "height": h, "pixels": h * w}
        for key in methods:
            finder = SEAM_FINDERS[key]
            row[f"{key}_find_ms"] = median_ms(lambda: finder(energy), repeats)
            row[f"{key}_seam_ms"] = median_ms(
                lambda: resize(img, h, w - seams, finder=finder, energy_fn=energy_fn),
                repeats) / seams
        rows.append(row)
        print(f"  {w:>4}x{h:<4} find ms: " +
              ", ".join(f"{k} {row[f'{k}_find_ms']:.3f}" for k in methods))
    return rows


# ---------------------------------------------------------------- report

def summary_rows(runs: dict) -> list:
    """One row per method: times and seam energies, with ratios against DP."""
    dp = runs.get("dp")
    out = []
    for key, r in runs.items():
        k = len(r["costs"])
        row = {"method": key, "total_s": r["total_s"], "finder_s": r["finder_s"],
               "finder_ms_per_seam": 1000 * r["finder_s"] / k,
               "first_cost": r["costs"][0], "mean_cost": statistics.mean(r["costs"]),
               "total_cost": sum(r["costs"])}
        if dp is not None:
            ratios = np.array(r["costs"]) / np.maximum(np.array(dp["costs"]), 1e-12)
            row.update(energy_vs_dp=row["total_cost"] / sum(dp["costs"]),
                       median_ratio=float(np.median(ratios)), max_ratio=float(ratios.max()),
                       find_speedup=dp["finder_s"] / r["finder_s"],
                       total_speedup=dp["total_s"] / r["total_s"])
        out.append(row)
    return out


def write_markdown(path: Path, img_shape, new_width, energy_name, runs, rows):
    h, w = img_shape[:2]
    k = w - new_width
    lines = [
        f"## Greedy variants vs dynamic programming ({w}x{h} -> {new_width}x{h}, "
        f"{k} vertical seams, {energy_name} energy)",
        "",
        "| Method | Total time (s) | Finder only (s) | Finder ms / seam | First seam energy | "
        "Mean seam energy | Energy vs DP | Median / max per-seam ratio | Find speed-up | "
        "Total speed-up |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary_rows(runs):
        lines.append(
            f"| {NAMES[s['method']]} | {s['total_s']:.2f} | {s['finder_s']:.3f} | "
            f"{s['finder_ms_per_seam']:.3f} | {s['first_cost']:,.0f} | {s['mean_cost']:,.0f} | "
            f"{s.get('energy_vs_dp', float('nan')):.2f}x | "
            f"{s.get('median_ratio', float('nan')):.2f}x / {s.get('max_ratio', float('nan')):.2f}x | "
            f"{s.get('find_speedup', float('nan')):.1f}x | {s.get('total_speedup', float('nan')):.2f}x |"
        )
    lines += [
        "",
        "Energy vs DP = total energy removed by the method / total energy removed by DP "
        "(1.00x = as good as optimal). Speed-ups are DP time / method time.",
        "",
        "### Time to find one seam vs image size (ms, median of repeats)",
        "",
        "| Size | Pixels | " + " | ".join(NAMES[key] for key in runs) + " |",
        "|---|---:|" + "---:|" * len(runs),
    ]
    for r in rows:
        lines.append(f"| {r['width']}x{r['height']} | {r['pixels']:,} | " +
                     " | ".join(f"{r[f'{key}_find_ms']:.3f}" for key in runs) + " |")
    lines += [
        "",
        "### Time for one full seam removal (energy + find + remove) vs image size (ms)",
        "",
        "| Size | " + " | ".join(NAMES[key] for key in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    for r in rows:
        lines.append(f"| {r['width']}x{r['height']} | " +
                     " | ".join(f"{r[f'{key}_seam_ms']:.2f}" for key in runs) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare greedy seam finding with DP.")
    parser.add_argument("image", type=Path, nargs="?", default=Path("samples/broadway_tower.jpg"))
    parser.add_argument("--out", type=Path, default=Path("out/greedy"))
    parser.add_argument("--width", type=int, help="target width (default: 70%% of the input)")
    parser.add_argument("--seams-shown", type=int, default=40,
                        help="how many removed seams to draw in seams_all.png")
    parser.add_argument("--energy", choices=sorted(ENERGY_FUNCTIONS), default="dual-gradient")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                        help="methods to compare (default: all; dp and greedy are always "
                             "included)")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.85, 1.0],
                        help="sizes for the timing table, as fractions of the input")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-charts", action="store_true", help="skip the matplotlib charts")
    args = parser.parse_args(argv)

    source = Image.open(args.image).convert("RGB")
    img = np.asarray(source)
    h, w = img.shape[:2]
    new_width = args.width if args.width is not None else round(0.7 * w)
    if not 1 <= new_width < w:
        parser.error(f"--width must be between 1 and {w - 1}")
    energy_fn = ENERGY_FUNCTIONS[args.energy]
    methods = [m for m in METHODS if m in set(args.methods) | {"dp", "greedy"}]
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Carve with every method.
    runs = {}
    for key in methods:
        runs[key] = carve_tracked(img, new_width, SEAM_FINDERS[key], energy_fn)
        print(f"{NAMES[key]:>24}: {w}x{h} -> {new_width}x{h} in {runs[key]['total_s']:.2f} s "
              f"(finder {runs[key]['finder_s']:.3f} s)")

    # 2. Seam pictures.
    first = draw_seams(img, [runs["dp"]["seams"][0]], DP_COLOR, thickness=3)
    first = draw_seams(first, [runs["greedy"]["seams"][0]], GREEDY_COLOR, thickness=3)
    Image.fromarray(first).save(args.out / "seams_first.png")

    shown = min(args.seams_shown, w - new_width)
    panels([(f"{NAMES[key]}: first {shown} seams",
             Image.fromarray(draw_seams(img, runs[key]["seams"][:shown], DP_COLOR)))
            for key in methods]).save(args.out / "seams_all.png")

    # 3. Visual quality.
    panels([(f"Original {w}x{h}", source)] +
           [(f"{NAMES[key]} {new_width}x{h} ({runs[key]['total_s']:.1f} s)",
             Image.fromarray(runs[key]["image"])) for key in methods],
           columns=len(methods) + 1).save(args.out / "result.png")

    # 4. Timing against image size.
    print("Timing against image size:")
    rows = timing_rows(source, args.scales, energy_fn, args.repeats, methods)

    write_markdown(args.out / "greedy_vs_dp.md", img.shape, new_width, args.energy, runs, rows)
    if not args.no_charts:
        seam_cost_chart(args.out / "seam_cost.png", {k: runs[k]["costs"] for k in methods})
        _line_chart(
            args.out / "timing.png",
            "Time to find one seam vs image size",
            "Milliseconds per seam search",
            [r["pixels"] / 1e6 for r in rows],
            [(NAMES[k], [r[f"{k}_find_ms"] for r in rows], COLORS[k]) for k in methods],
        )
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
