"""Compare graph cut (min-cut / max-flow) seam finding with DP.

Usage:
    python graphcut_compare.py [IMAGE] [--out out/graphcut] [--max-side 300]
                               [--seams 40] [--energy dual-gradient|sobel|forward]
                               [--scales ...] [--repeats 3] [--no-charts]

Graph cut is about 100-400x slower than DP, so the image is first shrunk so its
longer side is at most --max-side pixels (0 = keep the full size), and --seams
seams are removed in each direction. Both methods use the same energy function
and the same carving loop (carve_tracked from greedy_compare.py), so the only
difference is how the seam is chosen.

Horizontal seams: the image is transposed, carved with vertical seams and
transposed back, exactly like resize() does.

Writes to the output folder:
    seams_vertical.png / seams_horizontal.png   seams removed by DP and by graph cut
    result.png                 original, DP result and graph cut result
    graphcut_vs_dp.md          all the numbers as Markdown tables (paste into the report)
    timing.png                 time to find one seam vs image size
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark import _line_chart, median_ms
from greedy_compare import DP_COLOR, carve_tracked, draw_seams, panels
from seamcarve.dp import find_vertical_seam
from seamcarve.energy import ENERGY_FUNCTIONS
from seamcarve.graphcut import find_vertical_seam_graphcut

GC_COLOR = (20, 170, 60)        # green
FINDERS = {
    "dp": find_vertical_seam,
    "graphcut": find_vertical_seam_graphcut,                                  # Dinic
    "graphcut-ek": lambda e: find_vertical_seam_graphcut(e, "edmonds_karp"),  # Edmonds-Karp
}
NAMES = {"dp": "Dynamic programming", "graphcut": "Graph cut (Dinic)",
         "graphcut-ek": "Graph cut (Edmonds-Karp)"}
COLORS = {"dp": "#2a78d6", "graphcut": "#1d9e75", "graphcut-ek": "#eb6834"}


def carve_both(img, seams, energy_fn, direction):
    """Run DP and graph cut on the same image; horizontal = carve the transpose."""
    work = img if direction == "vertical" else np.swapaxes(img, 0, 1)
    target = work.shape[1] - seams
    return {key: carve_tracked(work, target, FINDERS[key], energy_fn) for key in ("dp", "graphcut")}


def compare_runs(runs) -> dict:
    dp, gc = runs["dp"], runs["graphcut"]
    k = len(dp["costs"])
    return {
        "seams": k,
        "dp_ms": 1000 * dp["finder_s"] / k,
        "gc_ms": 1000 * gc["finder_s"] / k,
        "dp_cost": sum(dp["costs"]),
        "gc_cost": sum(gc["costs"]),
        "same_cost": sum(np.isclose(a, b) for a, b in zip(dp["costs"], gc["costs"])),
        "same_seam": sum(np.array_equal(a, b) for a, b in zip(dp["seams"], gc["seams"])),
        "same_image": np.array_equal(dp["image"], gc["image"]),
    }


def timing_rows(source: Image.Image, scales, energy_fn, repeats: int):
    """Median time to find ONE vertical seam, per method, for several image sizes."""
    rows = []
    for scale in sorted(scales):
        size = (max(3, round(source.width * scale)), max(3, round(source.height * scale)))
        energy = energy_fn(np.asarray(source.resize(size, Image.Resampling.LANCZOS)))
        h, w = energy.shape
        row = {"width": w, "height": h, "pixels": h * w}
        for key, finder in FINDERS.items():
            # Edmonds-Karp gets very slow on big images: only time it up to 40k pixels.
            if key == "graphcut-ek" and h * w > 40_000:
                row[key] = float("nan")
                continue
            row[key] = median_ms(lambda: finder(energy), repeats)
        rows.append(row)
        print(f"  {w:>4}x{h:<4} " + ", ".join(f"{k} {row[k]:.1f} ms" for k in FINDERS))
    return rows


def write_markdown(path: Path, shape, energy_name, results, rows):
    h, w = shape[:2]
    lines = [
        f"## Graph cut vs dynamic programming ({w}x{h}, {energy_name} energy)",
        "",
        "| Direction | Seams | DP ms / seam | Graph cut ms / seam | Slower by | "
        "Total energy DP | Total energy graph cut | Same seam energy | Same seam | Same result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for direction, r in results.items():
        lines.append(
            f"| {direction} | {r['seams']} | {r['dp_ms']:.2f} | {r['gc_ms']:.1f} | "
            f"{r['gc_ms'] / r['dp_ms']:.0f}x | {r['dp_cost']:,.0f} | {r['gc_cost']:,.0f} | "
            f"{r['same_cost']}/{r['seams']} | {r['same_seam']}/{r['seams']} | "
            f"{'yes' if r['same_image'] else 'no'} |")
    lines += [
        "",
        "Same seam energy: both find a minimum seam, so the energies must match. "
        "The seams themselves can only differ when two seams tie.",
        "",
        "### Time to find one vertical seam vs image size (ms, median of repeats)",
        "",
        "| Size | Pixels | " + " | ".join(NAMES[k] for k in FINDERS) + " |",
        "|---|---:|" + "---:|" * len(FINDERS),
    ]
    for r in rows:
        lines.append(f"| {r['width']}x{r['height']} | {r['pixels']:,} | " +
                     " | ".join("-" if np.isnan(r[k]) else f"{r[k]:.1f}" for k in FINDERS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare graph cut seam finding with DP.")
    parser.add_argument("image", type=Path, nargs="?", default=Path("samples/broadway_tower.jpg"))
    parser.add_argument("--out", type=Path, default=Path("out/graphcut"))
    parser.add_argument("--max-side", type=int, default=300,
                        help="shrink so the longer side is at most this (0 = full size)")
    parser.add_argument("--seams", type=int, default=40, help="seams to remove per direction")
    parser.add_argument("--energy", choices=sorted(ENERGY_FUNCTIONS), default="dual-gradient")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.15, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args(argv)

    source = Image.open(args.image).convert("RGB")
    if args.max_side and max(source.size) > args.max_side:
        f = args.max_side / max(source.size)
        source = source.resize((round(source.width * f), round(source.height * f)),
                               Image.Resampling.LANCZOS)
    img = np.asarray(source)
    h, w = img.shape[:2]
    if not 1 <= args.seams < min(h, w):
        parser.error(f"--seams must be between 1 and {min(h, w) - 1}")
    energy_fn = ENERGY_FUNCTIONS[args.energy]
    args.out.mkdir(parents=True, exist_ok=True)

    results, carved = {}, {}
    for direction in ("vertical", "horizontal"):
        start = time.perf_counter()
        runs = carve_both(img, args.seams, energy_fn, direction)
        results[direction] = compare_runs(runs)
        r = results[direction]
        print(f"{direction:>10}: DP {r['dp_ms']:.2f} ms/seam, graph cut {r['gc_ms']:.1f} ms/seam, "
              f"same energy {r['same_cost']}/{r['seams']}, same seam {r['same_seam']}/{r['seams']} "
              f"({time.perf_counter() - start:.1f} s)")

        # Draw the seams on the (un-transposed) original.
        work = img if direction == "vertical" else np.swapaxes(img, 0, 1)
        pics = [(f"{NAMES[k]}: {direction} seams",
                 draw_seams(work, runs[k]["seams"], DP_COLOR if k == "dp" else GC_COLOR))
                for k in ("dp", "graphcut")]
        if direction == "horizontal":
            pics = [(label, np.swapaxes(p, 0, 1)) for label, p in pics]
        panels([(label, Image.fromarray(np.ascontiguousarray(p))) for label, p in pics]
               ).save(args.out / f"seams_{direction}.png")
        carved[direction] = runs

    panels([(f"Original {w}x{h}", source)] +
           [(f"{NAMES[k]} {w - args.seams}x{h}", Image.fromarray(carved["vertical"][k]["image"]))
            for k in ("dp", "graphcut")], columns=3).save(args.out / "result.png")

    print("Timing against image size:")
    rows = timing_rows(source, args.scales, energy_fn, args.repeats)
    write_markdown(args.out / "graphcut_vs_dp.md", img.shape, args.energy, results, rows)
    if not args.no_charts:
        _line_chart(
            args.out / "timing.png",
            "Time to find one seam vs image size",
            "Milliseconds per seam search",
            [r["pixels"] / 1e6 for r in rows],
            [(NAMES[k], [r[k] for r in rows], COLORS[k]) for k in FINDERS],
        )
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
