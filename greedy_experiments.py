"""Question 3: repeat the greedy vs DP comparison on several images and energy functions.

Usage:
    python greedy_experiments.py [--images samples/broadway_tower.jpg samples/scene.png]
                                 [--energies dual-gradient sobel] [--shrink 0.3]
                                 [--out out/greedy/experiments]

greedy_compare.py studies ONE image in detail. This script checks that its
conclusions hold in general: for every (image, energy function) pair it removes
the same number of vertical seams (--shrink, default 30% of the width) with every
method and records time and removed energy relative to DP.

Writes to the output folder:
    experiments.csv              one row per (image, energy, method)
    experiments.md               the same as Markdown tables (paste into the report)
    result_<image>_<energy>.png  original and the result of each method
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from greedy_compare import METHODS, NAMES, panels, run_methods, summary_rows
from seamcarve.energy import ENERGY_FUNCTIONS

FIELDS = ["image", "size", "energy", "method", "seams", "total_s", "finder_ms_per_seam",
          "mean_cost", "energy_vs_dp", "median_ratio", "max_ratio", "find_speedup",
          "total_speedup"]


def run(images, energies, shrink: float, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for path in images:
        source = Image.open(path).convert("RGB")
        img = np.asarray(source)
        h, w = img.shape[:2]
        new_width = max(1, round((1 - shrink) * w))
        for energy_name in energies:
            print(f"{path.name} {w}x{h} -> {new_width}x{h}, {energy_name}")
            runs = run_methods(img, new_width, ENERGY_FUNCTIONS[energy_name], METHODS)
            for s in summary_rows(runs):
                table.append({"image": path.name, "size": f"{w}x{h}", "energy": energy_name,
                              "seams": w - new_width, **{k: s[k] for k in FIELDS if k in s}})
                print(f"  {NAMES[s['method']]:>24}: {s['total_s']:6.2f} s, "
                      f"energy {s['energy_vs_dp']:.2f}x DP")
            panels([(f"Original {w}x{h}", source)] +
                   [(f"{NAMES[k]} ({runs[k]['total_s']:.1f} s)", Image.fromarray(runs[k]["image"]))
                    for k in METHODS], columns=len(METHODS) + 1
                   ).save(out_dir / f"result_{path.stem}_{energy_name}.png")
    return table


def write_csv(table, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in table:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def write_markdown(table, path: Path, shrink: float) -> None:
    lines = [
        f"## Greedy variants vs DP on every image and energy function "
        f"(width reduced by {shrink:.0%})",
        "",
        "| Image | Energy | Method | Total time (s) | Finder ms / seam | Energy vs DP | "
        "Median / max per-seam ratio | Find speed-up | Total speed-up |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(
            f"| {r['image']} ({r['size']}) | {r['energy']} | {NAMES[r['method']]} | "
            f"{r['total_s']:.2f} | {r['finder_ms_per_seam']:.3f} | {r['energy_vs_dp']:.2f}x | "
            f"{r['median_ratio']:.2f}x / {r['max_ratio']:.2f}x | {r['find_speedup']:.1f}x | "
            f"{r['total_speedup']:.2f}x |")
    lines += ["", "Energy vs DP = total energy removed / total energy removed by DP "
              "(1.00x = optimal). Speed-ups are DP time / method time."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Greedy vs DP on several images and energies.")
    parser.add_argument("--images", type=Path, nargs="+",
                        default=[Path("samples/broadway_tower.jpg"), Path("samples/scene.png")])
    parser.add_argument("--energies", nargs="+", choices=sorted(ENERGY_FUNCTIONS),
                        default=["dual-gradient", "sobel"])
    parser.add_argument("--shrink", type=float, default=0.3,
                        help="fraction of the width to remove (default 0.3)")
    parser.add_argument("--out", type=Path, default=Path("out/greedy/experiments"))
    args = parser.parse_args(argv)
    if not 0 < args.shrink < 1:
        parser.error("--shrink must be between 0 and 1")

    table = run(args.images, args.energies, args.shrink, args.out)
    write_csv(table, args.out / "experiments.csv")
    write_markdown(table, args.out / "experiments.md", args.shrink)
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
