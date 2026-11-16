"""Benchmark: measure running time and memory against image size.

Checks the complexity claims empirically:
    * energy map, one DP seam search, and one full seam removal are all O(m·n),
      so time / (m·n) should stay roughly constant as the image grows;
    * the low-memory DP needs about 1 byte per pixel instead of about 9.

Usage:
    python benchmark.py [--image samples/broadway_tower.jpg] [--out out/benchmark]
                        [--seams 20] [--repeats 5] [--scales 0.3 0.5 0.7 0.85 1.0]

Writes to the output folder:
    benchmark.csv        one row per image size, all measurements
    benchmark.md         the same as a Markdown table (paste into the report)
    time_per_seam.png    time to remove one seam vs number of pixels (default vs --fast)
    dp_memory.png        peak memory of one DP seam search, bytes per pixel
"""

import argparse
import csv
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image

from seamcarve.carve import resize
from seamcarve.dp import find_vertical_seam, find_vertical_seam_low_memory
from seamcarve.energy import dual_gradient_energy

COLUMNS = [
    "width", "height", "pixels",
    "energy_ms", "dp_ms", "dp_low_memory_ms",
    "seam_default_ms", "seam_fast_ms", "speedup",
    "dp_bytes_per_pixel", "dp_low_memory_bytes_per_pixel",
    "seam_default_ns_per_pixel",
]


def median_ms(fn, repeats: int) -> float:
    """Median wall-clock time of fn() over `repeats` runs, in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return 1000 * statistics.median(times)


def peak_bytes(fn) -> int:
    """Peak extra memory allocated while fn() runs (Python + NumPy buffers)."""
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def measure(img: np.ndarray, seams: int, repeats: int) -> dict:
    h, w = img.shape[:2]
    pixels = h * w
    energy = dual_gradient_energy(img)
    target_w = w - seams

    energy_ms = median_ms(lambda: dual_gradient_energy(img), repeats)
    dp_ms = median_ms(lambda: find_vertical_seam(energy), repeats)
    dp_low_ms = median_ms(lambda: find_vertical_seam_low_memory(energy), repeats)

    # Full pipeline (energy + DP + removal), averaged over `seams` removals.
    default_ms = median_ms(lambda: resize(img, h, target_w), repeats) / seams
    fast_ms = median_ms(
        lambda: resize(img, h, target_w, finder=find_vertical_seam_low_memory,
                       incremental=True), repeats) / seams

    return {
        "width": w, "height": h, "pixels": pixels,
        "energy_ms": energy_ms, "dp_ms": dp_ms, "dp_low_memory_ms": dp_low_ms,
        "seam_default_ms": default_ms, "seam_fast_ms": fast_ms,
        "speedup": default_ms / fast_ms,
        "dp_bytes_per_pixel": peak_bytes(lambda: find_vertical_seam(energy)) / pixels,
        "dp_low_memory_bytes_per_pixel":
            peak_bytes(lambda: find_vertical_seam_low_memory(energy)) / pixels,
        "seam_default_ns_per_pixel": 1e6 * default_ms / pixels,
    }


def linear_fit_r2(x, y) -> tuple[float, float]:
    """Least-squares fit y = a·x + b; returns (a, R²). R² near 1 means linear growth."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    residual = np.sum((y - (a * x + b)) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return a, (1 - residual / total) if total > 0 else 1.0


def write_csv(rows, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def write_markdown(rows, path: Path, seams: int) -> None:
    _, r2_default = linear_fit_r2([r["pixels"] for r in rows], [r["seam_default_ms"] for r in rows])
    _, r2_dp = linear_fit_r2([r["pixels"] for r in rows], [r["dp_ms"] for r in rows])
    lines = [
        f"Benchmark (dual-gradient energy, {seams} vertical seams per size; times are medians)",
        "",
        "| Size | Pixels | Energy (ms) | DP (ms) | DP low-mem (ms) | Seam default (ms) "
        "| Seam --fast (ms) | Speed-up | DP bytes/px | DP low-mem bytes/px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['width']}x{r['height']} | {r['pixels']:,} | {r['energy_ms']:.2f} | "
            f"{r['dp_ms']:.2f} | {r['dp_low_memory_ms']:.2f} | {r['seam_default_ms']:.2f} | "
            f"{r['seam_fast_ms']:.2f} | {r['speedup']:.2f}x | {r['dp_bytes_per_pixel']:.1f} | "
            f"{r['dp_low_memory_bytes_per_pixel']:.1f} |"
        )
    lines += [
        "",
        f"Linear fit of time against pixels (m·n): R² = {r2_default:.3f} for one full seam "
        f"removal, R² = {r2_dp:.3f} for one DP seam search. "
        "R² close to 1 supports the O(m·n) per-seam analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Chart styling: validated categorical slots 1 and 2 (light mode), recessive grid,
# text in text colours (never the series colour), legend + direct labels.
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"


def _line_chart(path, title, ylabel, x, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for label, y, color in series:
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=28)
    ax.set_xlabel("Image size (megapixels, m·n)", fontsize=9, color=TEXT_2)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)
    ax.set_ylim(0, 1.12 * max(max(y) for _, y, _ in series))
    ax.set_xlim(left=0, right=max(x) * 1.25)          # room for the direct labels
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    # Legend in one row between the title and the plot, so it never covers data.
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=len(series), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_charts(rows, out_dir: Path) -> None:
    mp = [r["pixels"] / 1e6 for r in rows]
    _line_chart(
        out_dir / "time_per_seam.png",
        "Time to remove one seam grows linearly with image size",
        "Milliseconds per seam",
        mp,
        [("Default", [r["seam_default_ms"] for r in rows], SERIES_1),
         ("--fast", [r["seam_fast_ms"] for r in rows], SERIES_2)],
    )
    _line_chart(
        out_dir / "dp_memory.png",
        "Peak memory of one DP seam search",
        "Bytes per pixel",
        mp,
        [("Full table", [r["dp_bytes_per_pixel"] for r in rows], SERIES_1),
         ("Low-memory", [r["dp_low_memory_bytes_per_pixel"] for r in rows], SERIES_2)],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark seam carving against image size.")
    parser.add_argument("--image", type=Path, default=Path("samples/broadway_tower.jpg"))
    parser.add_argument("--out", type=Path, default=Path("out/benchmark"))
    parser.add_argument("--seams", type=int, default=20, help="seams removed per size")
    parser.add_argument("--repeats", type=int, default=5, help="repeats for the timings")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.3, 0.45, 0.6, 0.75, 0.9, 1.0],
                        help="sizes to test, as fractions of the input image")
    parser.add_argument("--no-charts", action="store_true", help="skip the PNG charts")
    args = parser.parse_args(argv)

    source = Image.open(args.image).convert("RGB")
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for scale in sorted(args.scales):
        size = (max(args.seams + 2, round(source.width * scale)), max(3, round(source.height * scale)))
        img = np.asarray(source.resize(size, Image.Resampling.LANCZOS))
        row = measure(img, args.seams, args.repeats)
        rows.append(row)
        print(f"{row['width']:>4}x{row['height']:<4}  seam: default {row['seam_default_ms']:7.2f} ms"
              f"  fast {row['seam_fast_ms']:7.2f} ms  ({row['speedup']:.2f}x)   DP memory "
              f"{row['dp_bytes_per_pixel']:.1f} -> {row['dp_low_memory_bytes_per_pixel']:.1f} B/px")

    write_csv(rows, args.out / "benchmark.csv")
    write_markdown(rows, args.out / "benchmark.md", args.seams)
    if not args.no_charts:
        write_charts(rows, args.out)
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
