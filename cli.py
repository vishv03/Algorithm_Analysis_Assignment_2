"""Command-line seam carving resizer.

Usage:
    python cli.py INPUT OUTPUT [--width W] [--height H]

Examples:
    python cli.py samples/scene.png out/scene_w300.png --width 300
    python cli.py samples/scene.png out/scene_small.png --width 300 --height 200

Omitting --width or --height keeps that dimension unchanged.
Pillow is used only to read and write image files; all the seam carving
(energy, DP, seam removal) is our own code in the seamcarve package.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from seamcarve.carve import SEAM_FINDERS, resize
from seamcarve.dp import find_vertical_seam_low_memory
from seamcarve.energy import ENERGY_FUNCTIONS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shrink an image with seam carving (dynamic programming)."
    )
    parser.add_argument("input", type=Path, help="input image file (jpg, png, ...)")
    parser.add_argument("output", type=Path, help="where to write the resized image")
    parser.add_argument("--width", type=int, help="target width in pixels (default: unchanged)")
    parser.add_argument("--height", type=int, help="target height in pixels (default: unchanged)")
    parser.add_argument("--energy", choices=sorted(ENERGY_FUNCTIONS), default="dual-gradient",
                        help="energy function (default: dual-gradient)")
    parser.add_argument("--fast", action="store_true",
                        help="use the improvements: local energy update + low-memory DP "
                             "(same result, faster)")
    parser.add_argument("--finder", choices=sorted(SEAM_FINDERS), default="dp",
                        help="seam finding algorithm: dp (optimal, Q1), greedy "
                             "(row-by-row local choice, Q3), or graphcut "
                             "(min-cut / max-flow, optimal but slow) (default: dp)")
    return parser.parse_args(argv)


def choose_finder(name: str, fast: bool):
    """The seam finder for --finder, using the low-memory DP when --fast is on."""
    if name == "dp" and fast:
        return find_vertical_seam_low_memory
    return SEAM_FINDERS[name]


def main(argv=None) -> int:
    args = parse_args(argv)

    # Load as RGB so the array is always (height, width, 3) uint8,
    # even for grayscale, palette or RGBA input files.
    img = np.asarray(Image.open(args.input).convert("RGB"))
    height, width = img.shape[:2]
    new_width = args.width if args.width is not None else width
    new_height = args.height if args.height is not None else height

    print(f"Input:  {args.input}  {width} x {height}")
    print(f"Target: {new_width} x {new_height}  "
          f"({width - new_width} vertical + {height - new_height} horizontal seams), "
          f"energy: {args.energy}, finder: {args.finder}")

    start = time.perf_counter()
    try:
        out = resize(
            img, new_height, new_width,
            energy_fn=ENERGY_FUNCTIONS[args.energy],
            finder=choose_finder(args.finder, args.fast),
            incremental=args.fast,
        )
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.output)
    print(f"Output: {args.output}  {out.shape[1]} x {out.shape[0]}  in {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
