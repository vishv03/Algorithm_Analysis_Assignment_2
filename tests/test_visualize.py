"""Tests for the bonus features: carve_steps, seam tracking, removal-order maps,
the step-by-step visualisation script and the interactive server."""

import json
import threading
import urllib.request

import numpy as np
import pytest
from PIL import Image

from seamcarve.carve import carve_steps, resize
from seamcarve.dp import find_vertical_seam_low_memory
from seamcarve.energy import dual_gradient_energy, sobel_energy
from seamcarve.visualize import (
    all_seams_overlay,
    apply_removal_order,
    energy_to_image,
    removal_order,
    track_seams,
)


def random_image(m=14, n=18, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (m, n, 3), dtype=np.uint8)


# ---------------------------------------------------------------- carve_steps

@pytest.mark.parametrize("incremental", [False, True])
def test_carve_steps_ends_with_the_resize_result(incremental):
    img = random_image()
    steps = list(carve_steps(img, 10, 12, incremental=incremental))
    assert len(steps) == (18 - 12) + (14 - 10)
    assert [s.number for s in steps] == list(range(1, len(steps) + 1))
    np.testing.assert_array_equal(steps[-1].result, resize(img, 10, 12))


def test_carve_steps_alternates_directions_then_finishes_the_rest():
    steps = list(carve_steps(random_image(), 12, 12))      # 6 vertical, 2 horizontal
    directions = [s.direction for s in steps]
    assert directions[:4] == ["vertical", "horizontal", "vertical", "horizontal"]
    assert directions[4:] == ["vertical"] * 4


@pytest.mark.parametrize("incremental", [False, True])
def test_step_energy_and_cost_match_the_image(incremental):
    for step in carve_steps(random_image(), 10, 13, incremental=incremental):
        np.testing.assert_allclose(step.energy, dual_gradient_energy(step.image))
        if step.direction == "vertical":
            expected = step.energy[np.arange(step.energy.shape[0]), step.seam].sum()
        else:
            expected = step.energy[step.seam, np.arange(step.energy.shape[1])].sum()
        assert step.cost == pytest.approx(expected)


def test_carve_steps_validates_target():
    with pytest.raises(ValueError):
        next(carve_steps(random_image(), 20, 5))


# ---------------------------------------------------------------- track_seams

def test_tracked_seams_cover_exactly_the_removed_pixels():
    """Removing the tracked original pixels gives the same image as carving."""
    img = np.arange(9 * 11 * 3).reshape(9, 11, 3)          # every pixel unique
    removed = np.zeros((9, 11), dtype=bool)
    result = img
    for step, rows, cols in track_seams(img, 6, 7):
        assert not removed[rows, cols].any()               # each pixel removed once
        removed[rows, cols] = True
        # the seam pixels in the current image are those original pixels
        if step.direction == "vertical":
            current = step.image[np.arange(step.image.shape[0]), step.seam]
        else:
            current = step.image[step.seam, np.arange(step.image.shape[1])]
        np.testing.assert_array_equal(current, img[rows, cols])
        result = step.result
    assert removed.sum() == 9 * 11 - 6 * 7
    # the kept pixels, in order, are the result
    assert sorted(img[~removed][:, 0].tolist()) == sorted(result[..., 0].ravel().tolist())


# ---------------------------------------------------------------- removal-order maps

@pytest.mark.parametrize("energy_fn", [dual_gradient_energy, sobel_energy])
def test_vertical_order_map_reproduces_every_width(energy_fn):
    img = random_image(12, 16)
    order = removal_order(img, 4, "vertical", energy_fn=energy_fn)
    for w in range(4, 17):
        np.testing.assert_array_equal(apply_removal_order(img, order, w, "vertical"),
                                      resize(img, 12, w, energy_fn=energy_fn))


def test_horizontal_order_map_reproduces_every_height():
    img = random_image(13, 9)
    order = removal_order(img, 3, "horizontal")
    for h in range(3, 14):
        np.testing.assert_array_equal(apply_removal_order(img, order, h, "horizontal"),
                                      resize(img, h, 9))


def test_fast_mode_gives_the_same_order_map():
    img = random_image()
    slow = removal_order(img, 5, "vertical")
    fast = removal_order(img, 5, "vertical", finder=find_vertical_seam_low_memory,
                         incremental=True)
    np.testing.assert_array_equal(slow, fast)


def test_order_map_values_and_errors():
    img = random_image(6, 8)
    order = removal_order(img, 5)
    assert order.max() == 3                                # survivors carry the step count
    assert (np.sum(order < 3, axis=1) == 3).all()          # one pixel per row per step
    with pytest.raises(ValueError):
        apply_removal_order(img, order, 4)                 # below the precomputed minimum
    with pytest.raises(ValueError):
        removal_order(img, 5, "diagonal")


# ---------------------------------------------------------------- drawing helpers

def test_overlay_and_energy_image():
    img = random_image(8, 10)
    removed = [(s.direction, r, c) for s, r, c in track_seams(img, 7, 8)]
    canvas = all_seams_overlay(img, removed)
    assert canvas.shape == img.shape and canvas.dtype == np.uint8
    for direction, rows, cols in removed:
        expected_red = 255 if direction == "vertical" else 0
        assert (canvas[rows, cols, 0] == expected_red).all()
    gray = energy_to_image(dual_gradient_energy(img))
    assert gray.shape == (8, 10, 3) and gray.dtype == np.uint8
    assert energy_to_image(np.zeros((3, 4))).max() == 0     # flat image: no division by zero


# ---------------------------------------------------------------- visualize.py

def test_visualize_script_writes_all_outputs(tmp_path):
    from visualize import main

    src, out = tmp_path / "in.png", tmp_path / "vis"
    Image.fromarray(random_image(20, 24)).save(src)
    assert main([str(src), str(out), "--width", "18", "--height", "16", "--frames",
                 "--max-side", "0"]) == 0
    gif = Image.open(out / "steps.gif")
    assert gif.n_frames == 10 + 1                           # every step + final frame
    assert Image.open(out / "result.png").size == (18, 16)
    np.testing.assert_array_equal(np.asarray(Image.open(out / "result.png")),
                                  resize(random_image(20, 24), 16, 18))
    assert len(list((out / "frames").glob("step_*.png"))) == 10
    lines = (out / "seams.csv").read_text().splitlines()
    assert lines[0].startswith("step,direction") and len(lines) == 11
    assert (out / "summary.png").exists()


def test_visualize_script_rejects_bad_targets(tmp_path):
    from visualize import main

    src = tmp_path / "in.png"
    Image.fromarray(random_image(10, 10)).save(src)
    assert main([str(src), str(tmp_path / "a"), "--width", "20", "--max-side", "0"]) == 1
    assert main([str(src), str(tmp_path / "b"), "--max-side", "0"]) == 1   # nothing to do


# ---------------------------------------------------------------- interactive.py

@pytest.fixture
def server():
    from interactive import ResizeSession, make_server

    img = Image.fromarray(random_image(30, 40))
    session = ResizeSession(img, "test.png", preview_size=20, min_scale=0.5)
    srv = make_server(session, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}", session
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url) as res:
        return res.read(), res.headers


def test_interactive_state_and_horizontal_maps_are_exact(server):
    base, session = server
    state = json.loads(get(base + "/api/state")[0])
    n, m = state["preview"]
    assert (n, m) == (20, 15) and state["full"] == [40, 30] and state["min"] == [10, 8]
    import base64
    pixels = np.frombuffer(base64.b64decode(state["pixels"]), np.uint8).reshape(m, n, 3)
    vorder = np.frombuffer(base64.b64decode(state["vorder"]), "<u2").reshape(m, n)
    np.testing.assert_array_equal(apply_removal_order(pixels, vorder, 13), resize(pixels, m, 13))

    data = json.loads(get(base + "/api/horizontal?width=13")[0])
    carved = np.frombuffer(base64.b64decode(data["pixels"]), np.uint8).reshape(m, 13, 3)
    horder = np.frombuffer(base64.b64decode(data["horder"]), "<u2").reshape(m, 13)
    np.testing.assert_array_equal(apply_removal_order(carved, horder, 9, "horizontal"),
                                  resize(resize(pixels, m, 13), 9, 13))


def test_interactive_export_and_page(server):
    base, session = server
    png, headers = get(base + "/api/export?width=10&height=8")
    assert headers["X-Size"] == "20x16"                      # preview target scaled x2
    from io import BytesIO
    assert Image.open(BytesIO(png)).size == (20, 16)
    page, headers = get(base + "/")
    assert b"Interactive Seam Carving" in page and headers["Content-Type"].startswith("text/html")


def test_interactive_upload_and_errors(server):
    base, session = server
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(random_image(10, 12, seed=3)).save(buf, "PNG")
    req = urllib.request.Request(base + "/api/upload", data=buf.getvalue(),
                                 headers={"X-Filename": "new.png"}, method="POST")
    with urllib.request.urlopen(req) as res:
        state = json.loads(res.read())
    assert state["name"] == "new.png" and state["preview"] == [12, 10]

    bad = urllib.request.Request(base + "/api/upload", data=b"not an image", method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(bad)
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(base + "/api/horizontal")     # missing width
    assert err.value.code == 400
