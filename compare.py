"""Side-by-side comparison: original | scaling | cropping | seam carving.

Usage:
    python compare.py INPUT OUTPUT [--width W] [--height H]

Example:
    python compare.py samples/broadway_tower.jpg out/broadway_compare.jpg --width 640

Produces one image with four panels, each labelled:
    1. the original image
    2. plain scaling to the target size      (stretches/squashes content)
    3. a centred crop to the target size     (cuts content off the edges)
    4. our seam carving to the target size   (removes low-energy seams)

Scaling and cropping use Pillow, purely as baselines to compare against.
The seam-carved panel uses our own seamcarve.resize().
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from seamcarve.carve import resize

GAP = 10            # white space between panels, in pixels
LABEL_HEIGHT = 24   # space above each panel for its caption


def center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    left = (img.width - width) // 2
    top = (img.height - height) // 2
    return img.crop((left, top, left + width, top + height))


def build_comparison(original: Image.Image, width: int, height: int) -> tuple[Image.Image, float]:
    """Return the 4-panel comparison image and the seam carving time in seconds."""
    start = time.perf_counter()
    carved = Image.fromarray(resize(np.asarray(original), height, width))
    elapsed = time.perf_counter() - start

    panels = [
        ("Original", original),
        ("Scaling", original.resize((width, height), Image.Resampling.LANCZOS)),
        ("Cropping", center_crop(original, width, height)),
        (f"Seam carving ({elapsed:.1f} s)", carved),
    ]

    total_width = sum(p.width for _, p in panels) + GAP * (len(panels) - 1)
    canvas = Image.new("RGB", (total_width, original.height + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, panel in panels:
        draw.text((x + 4, 6), f"{label}  {panel.width}x{panel.height}", fill="black")
        canvas.paste(panel, (x, LABEL_HEIGHT))
        x += panel.width + GAP
    return canvas, elapsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare scaling, cropping and seam carving.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, help="target width (default: unchanged)")
    parser.add_argument("--height", type=int, help="target height (default: unchanged)")
    args = parser.parse_args(argv)

    original = Image.open(args.input).convert("RGB")
    width = args.width if args.width is not None else original.width
    height = args.height if args.height is not None else original.height

    try:
        canvas, elapsed = build_comparison(original, width, height)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"{original.width}x{original.height} -> {width}x{height}, "
          f"seam carving took {elapsed:.2f} s. Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
