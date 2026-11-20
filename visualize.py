"""Bonus (a): step-by-step visualisation of seam carving, with every seam shown.

Usage:
    python visualize.py INPUT [OUTDIR] [--width W] [--height H] [options]

Example:
    python visualize.py samples/broadway_tower.jpg out/visualize --width 280 --height 220

Writes to OUTDIR (default out/visualize):
    steps.gif     animation, one frame per step (or every --every steps), three panels:
                    1. the current image with the seam about to be removed (red / blue)
                    2. the energy map of the current image with the same seam
                    3. the original image with ALL seams removed so far
    summary.png   original | every removed seam on the original | result
    seams.csv     step, direction, seam energy and image size after each step
    frames/       one PNG per step (only with --frames)

Vertical seams are drawn red -> yellow (first -> last), horizontal seams
blue -> cyan. Seams are found by the same code as resize() (carve_steps), so
the result panel is exactly what cli.py would produce for the same input.
By default the input is first scaled down to --max-side pixels (Pillow, for
speed and a small GIF); use --max-side 0 to work at full size.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cli import choose_finder
from seamcarve.carve import SEAM_FINDERS
from seamcarve.energy import ENERGY_FUNCTIONS
from seamcarve.visualize import (
    HORIZONTAL_COLORS,
    VERTICAL_COLORS,
    all_seams_overlay,
    draw_seam,
    energy_to_image,
    seam_color,
    track_seams,
)

GAP = 8               # pixels between panels
CAPTION = 22          # caption strip above the panels
BACKGROUND = (32, 32, 32)
MAX_GIF_FRAMES = 60   # --every 0 picks the step so the GIF has about this many frames


def load_image(path: Path, max_side: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.LANCZOS)
    return np.asarray(img)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:            # very old Pillow without scalable default font
        return ImageFont.load_default()


def compose(panels, labels, panel_size, caption: str) -> Image.Image:
    """Lay out equal-size panels side by side with a caption and labels."""
    h, w = panel_size
    label_h = 18
    width = len(panels) * w + (len(panels) - 1) * GAP
    out = Image.new("RGB", (width, CAPTION + label_h + h), BACKGROUND)
    draw = ImageDraw.Draw(out)
    draw.text((4, 3), caption, fill=(255, 255, 255), font=_font(14))
    for k, (panel, label) in enumerate(zip(panels, labels)):
        x = k * (w + GAP)
        draw.text((x + 2, CAPTION + 1), label, fill=(190, 190, 190), font=_font(12))
        out.paste(Image.fromarray(panel), (x, CAPTION + label_h))
    return out


def pad(img: np.ndarray, size) -> np.ndarray:
    """Place img in the top-left corner of a background canvas of `size`."""
    canvas = np.empty(size + (3,), dtype=np.uint8)
    canvas[:] = BACKGROUND
    canvas[: img.shape[0], : img.shape[1]] = img
    return canvas


def save_gif(frames, durations, path: Path) -> None:
    """Save an animated GIF. Each frame gets its own 128-colour palette without
    dithering: dithering noise makes GIF compression (LZW) many times larger."""
    paletted = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
                for f in frames]
    paletted[0].save(path, save_all=True, append_images=paletted[1:],
                     duration=durations, loop=0, optimize=False)


def run(args) -> int:
    img = load_image(args.input, args.max_side)
    m, n = img.shape[:2]
    new_width = args.width if args.width is not None else n
    new_height = args.height if args.height is not None else m
    total = (n - new_width) + (m - new_height)
    if not (1 <= new_width <= n and 1 <= new_height <= m):
        print(f"Error: target {new_width} x {new_height} must be between 1 x 1 and the "
              f"(scaled) input size {n} x {m}", file=sys.stderr)
        return 1
    if total == 0:
        print("Nothing to do: the target size equals the input size.", file=sys.stderr)
        return 1

    every = args.every or max(1, math.ceil(total / MAX_GIF_FRAMES))
    print(f"Input: {args.input} -> working size {n} x {m}, target {new_width} x {new_height} "
          f"({total} seams), energy {args.energy}, finder {args.finder}, frame every {every} step(s)")

    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.frames:
        (out_dir / "frames").mkdir(exist_ok=True)

    finder = choose_finder(args.finder, args.fast)
    steps = track_seams(img, new_height, new_width, finder,
                        ENERGY_FUNCTIONS[args.energy], incremental=args.fast)

    overlay = img.copy()           # original with the seams removed so far
    removed = []                   # (direction, rows, cols) for summary.png
    frames, rows_csv = [], []
    labels = ["current image + next seam", "energy map + seam", "all seams so far (original)"]
    count = {"vertical": n - new_width, "horizontal": m - new_height}
    seen = {"vertical": 0, "horizontal": 0}
    result = img
    start = time.perf_counter()

    for step, rows, cols in steps:
        colors = VERTICAL_COLORS if step.direction == "vertical" else HORIZONTAL_COLORS
        color = seam_color(seen[step.direction] / max(count[step.direction] - 1, 1), colors)
        seen[step.direction] += 1
        overlay[rows, cols] = color
        removed.append((step.direction, rows, cols))
        result = step.result
        cost = step.cost
        rows_csv.append((step.number, step.direction, f"{cost:.1f}",
                         step.result.shape[1], step.result.shape[0]))

        keep_frame = (step.number - 1) % every == 0 or step.number == total
        if keep_frame or args.frames:
            highlight = (255, 0, 0) if step.direction == "vertical" else (0, 90, 255)
            current = step.image.copy()
            draw_seam(current, step.seam, step.direction, highlight)
            energy = energy_to_image(step.energy)
            draw_seam(energy, step.seam, step.direction, highlight)
            caption = (f"step {step.number}/{total}  {step.direction} seam  energy {cost:,.0f}  "
                       f"{step.image.shape[1]}x{step.image.shape[0]} -> "
                       f"{step.result.shape[1]}x{step.result.shape[0]}")
            frame = compose([pad(current, (m, n)), pad(energy, (m, n)), overlay.copy()],
                            labels, (m, n), caption)
            if keep_frame:
                frames.append(frame)
            if args.frames:
                frame.save(out_dir / "frames" / f"step_{step.number:04d}.png")
    elapsed = time.perf_counter() - start

    # Last frame: the finished result, held for a while at the end of the loop.
    final = compose([pad(result, (m, n)), pad(energy_to_image(ENERGY_FUNCTIONS[args.energy](result)),
                                              (m, n)), overlay],
                    ["result", "energy map of result", "all removed seams (original)"], (m, n),
                    f"done: {total} seams removed, {n}x{m} -> {result.shape[1]}x{result.shape[0]}")
    frames.append(final)
    durations = [round(1000 / args.fps)] * (len(frames) - 1) + [2500]
    save_gif(frames, durations, out_dir / "steps.gif")

    summary = compose([img, all_seams_overlay(img, removed), pad(result, (m, n))],
                      ["original", "all removed seams", "seam carving result"], (m, n),
                      f"{args.input.name}: {n}x{m} -> {result.shape[1]}x{result.shape[0]}, "
                      f"{total} seams ({args.energy}, {args.finder})  "
                      f"vertical: red->yellow, horizontal: blue->cyan")
    summary.save(out_dir / "summary.png")
    Image.fromarray(result).save(out_dir / "result.png")

    with open(out_dir / "seams.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "direction", "seam_energy", "width_after", "height_after"])
        writer.writerows(rows_csv)

    print(f"Carved {total} seams in {elapsed:.2f} s (including frame drawing)")
    print(f"Wrote {out_dir / 'steps.gif'} ({len(frames)} frames), {out_dir / 'summary.png'}, "
          f"{out_dir / 'result.png'}, {out_dir / 'seams.csv'}"
          + (f", {out_dir / 'frames'}/" if args.frames else ""))
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-by-step seam carving visualisation.")
    parser.add_argument("input", type=Path, help="input image")
    parser.add_argument("outdir", type=Path, nargs="?", default=Path("out/visualize"),
                        help="output folder (default: out/visualize)")
    parser.add_argument("--width", type=int, help="target width of the WORKING image (after --max-side)")
    parser.add_argument("--height", type=int, help="target height of the WORKING image (after --max-side)")
    parser.add_argument("--max-side", type=int, default=320,
                        help="scale the input so its longer side is at most this (0 = full size)")
    parser.add_argument("--energy", choices=sorted(ENERGY_FUNCTIONS), default="dual-gradient")
    parser.add_argument("--finder", choices=sorted(SEAM_FINDERS), default="dp")
    parser.add_argument("--fast", action="store_true", help="local energy update + low-memory DP")
    parser.add_argument("--every", type=int, default=0,
                        help=f"one GIF frame every N steps (0 = auto, about {MAX_GIF_FRAMES} frames)")
    parser.add_argument("--fps", type=float, default=12, help="GIF frames per second")
    parser.add_argument("--frames", action="store_true", help="also save every step as a PNG")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
