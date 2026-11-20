"""Bonus (b): interactive seam carving, resize an image live in the browser.

Usage:
    python interactive.py [IMAGE] [--port 8000] [--preview-size 480] [--min-scale 0.25]

Then open http://localhost:8000 (it opens automatically unless --no-browser).
Drag the width / height sliders, or the handle at the corner of the result,
and the seam-carved image updates immediately. Other controls: show the
removed seams on the original, compare with plain scaling / cropping, animate
the carving, switch the energy function, open another image, and download the
result at full resolution.

Only the Python standard library is used for the server (http.server), so
there is nothing extra to install. The page is webapp/index.html.

How it can be instant: removal-order maps (see seamcarve/visualize.py)
    1. On load, carve vertical seams on the preview image down to the minimum
       width ONCE, recording the step at which every pixel is removed. Any
       width w is then "keep pixels with order >= n - w": a mask, done in the
       browser in a few milliseconds, identical to resize(img, m, w).
    2. For the height, the server computes a horizontal removal-order map of
       the width-w image (cached per width) and the browser masks it the same
       way. So the result for (w, h) is "vertical seams first, then
       horizontal", exactly resize(resize(img, m, w), h, w).
    3. "Download" repeats the same two resizes on the FULL-resolution image
       (scaled target), using the fast mode (local energy update + low-memory DP).

The work per interaction is O(m·n) masking instead of O(k·m·n) carving.
"""

import argparse
import base64
import io
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from seamcarve.carve import resize
from seamcarve.dp import find_vertical_seam_low_memory
from seamcarve.energy import ENERGY_FUNCTIONS
from seamcarve.visualize import apply_removal_order, removal_order

WEBAPP = Path(__file__).resolve().parent / "webapp"
MAX_UPLOAD = 40 * 1024 * 1024        # bytes


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


class ResizeSession:
    """The image being edited and its precomputed removal-order maps.

    All carving uses the fast mode (incremental energy + low-memory DP), which
    gives exactly the same seams as the default mode.
    """

    def __init__(self, image: Image.Image, name: str, preview_size: int = 480,
                 min_scale: float = 0.25, energy: str = "dual-gradient"):
        self.lock = threading.Lock()
        self.preview_size = preview_size
        self.min_scale = min_scale
        self.load(image, name, energy)

    # -- setup ---------------------------------------------------------------
    def load(self, image: Image.Image, name: str, energy: str) -> None:
        if energy not in ENERGY_FUNCTIONS:
            raise ValueError(f"unknown energy function {energy!r}")
        image = image.convert("RGB")
        full = np.asarray(image)
        preview = image
        if max(image.size) > self.preview_size:
            scale = self.preview_size / max(image.size)
            preview = image.resize((max(2, round(image.width * scale)),
                                    max(2, round(image.height * scale))), Image.LANCZOS)
        preview = np.asarray(preview)
        m, n = preview.shape[:2]

        start = time.perf_counter()
        min_w = max(1, round(n * self.min_scale))
        vorder = removal_order(preview, min_w, "vertical", **self._carve_options(energy))
        seconds = time.perf_counter() - start

        with self.lock:
            self.name, self.energy = name, energy
            self.full, self.preview = full, preview
            self.min_w, self.min_h = min_w, max(1, round(m * self.min_scale))
            self.vorder = vorder
            self.horizontal_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            self.vertical_seconds = seconds

    @staticmethod
    def _carve_options(energy: str) -> dict:
        return {"finder": find_vertical_seam_low_memory,
                "energy_fn": ENERGY_FUNCTIONS[energy], "incremental": True}

    # -- API -----------------------------------------------------------------
    def state(self) -> dict:
        """Everything the page needs to produce any width on its own."""
        m, n = self.preview.shape[:2]
        return {
            "name": self.name, "energy": self.energy,
            "energies": sorted(ENERGY_FUNCTIONS),
            "full": [self.full.shape[1], self.full.shape[0]],
            "preview": [n, m], "min": [self.min_w, self.min_h],
            "pixels": _b64(self.preview),                          # RGB, row-major
            "vorder": _b64(self.vorder.astype("<u2")),             # uint16 little-endian
            "seconds": round(self.vertical_seconds, 3),
        }

    def horizontal(self, width: int) -> dict:
        """The width-`width` image and its horizontal removal-order map."""
        m, n = self.preview.shape[:2]
        width = int(np.clip(width, self.min_w, n))
        with self.lock:
            cached = self.horizontal_cache.get(width)
        start = time.perf_counter()
        if cached is None:
            carved = apply_removal_order(self.preview, self.vorder, width, "vertical")
            horder = removal_order(carved, self.min_h, "horizontal",
                                   **self._carve_options(self.energy))
            cached = (carved, horder)
            with self.lock:
                if len(self.horizontal_cache) > 64:      # keep memory bounded
                    self.horizontal_cache.clear()
                self.horizontal_cache[width] = cached
        carved, horder = cached
        return {"width": width, "height": m, "pixels": _b64(carved),
                "horder": _b64(horder.astype("<u2")),
                "seconds": round(time.perf_counter() - start, 3)}

    def export(self, width: int, height: int) -> tuple[bytes, tuple[int, int]]:
        """Full-resolution PNG for a preview-size target (scaled up)."""
        m, n = self.preview.shape[:2]
        fm, fn = self.full.shape[:2]
        full_w = int(np.clip(round(width * fn / n), 1, fn))
        full_h = int(np.clip(round(height * fm / m), 1, fm))
        options = self._carve_options(self.energy)
        out = resize(self.full, fm, full_w, **options)          # vertical seams first,
        out = resize(out, full_h, full_w, **options)            # then horizontal (as the preview)
        buffer = io.BytesIO()
        Image.fromarray(out).save(buffer, "PNG")
        return buffer.getvalue(), (full_w, full_h)


class Handler(BaseHTTPRequestHandler):
    session: ResizeSession = None    # set by make_server()

    def log_message(self, fmt, *args):      # keep the console quiet
        pass

    def _send(self, status: int, body: bytes, content_type: str, extra=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _error(self, err: Exception, status: int = 400) -> None:
        self._json({"error": str(err)}, status)

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                path = WEBAPP / "index.html"
                self._send(200, path.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                energy = query.get("energy")
                if energy and energy != self.session.energy:
                    self.session.load(Image.fromarray(self.session.full), self.session.name, energy)
                self._json(self.session.state())
            elif url.path == "/api/horizontal":
                self._json(self.session.horizontal(int(query["width"])))
            elif url.path == "/api/export":
                png, (w, h) = self.session.export(int(query["width"]), int(query["height"]))
                stem = Path(self.session.name).stem
                self._send(200, png, "image/png", {
                    "Content-Disposition": f'attachment; filename="{stem}_{w}x{h}.png"',
                    "X-Size": f"{w}x{h}"})
            else:
                self._send(404, b"not found", "text/plain")
        except (KeyError, ValueError) as err:
            self._error(err)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path != "/api/upload":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD:
                raise ValueError(f"upload must be between 1 byte and {MAX_UPLOAD // 2**20} MB")
            data = self.rfile.read(length)
            name = Path(self.headers.get("X-Filename", "upload.png")).name
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
            except Exception as err:            # Pillow raises several error types
                raise ValueError(f"not a readable image: {err}") from err
            self.session.load(image, name, self.session.energy)
            self._json(self.session.state())
        except ValueError as err:
            self._error(err)


def make_server(session: ResizeSession, host: str = "127.0.0.1", port: int = 8000):
    handler = type("SessionHandler", (Handler,), {"session": session})
    return ThreadingHTTPServer((host, port), handler)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive seam carving in the browser.")
    parser.add_argument("image", type=Path, nargs="?", default=Path("samples/broadway_tower.jpg"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default: this computer only)")
    parser.add_argument("--preview-size", type=int, default=480,
                        help="longer side of the live preview in pixels (default 480)")
    parser.add_argument("--min-scale", type=float, default=0.25,
                        help="smallest size offered, as a fraction of the original (default 0.25)")
    parser.add_argument("--energy", choices=sorted(ENERGY_FUNCTIONS), default="dual-gradient")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0 < args.min_scale < 1:
        print("Error: --min-scale must be between 0 and 1", file=sys.stderr)
        return 1
    print(f"Precomputing seams for {args.image} ...")
    session = ResizeSession(Image.open(args.image), args.image.name,
                            args.preview_size, args.min_scale, args.energy)
    server = make_server(session, args.host, args.port)
    url = f"http://{'localhost' if args.host == '127.0.0.1' else args.host}:{server.server_port}/"
    print(f"Ready ({session.vertical_seconds:.2f} s). Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
