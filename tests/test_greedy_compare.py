"""Tests for the greedy vs DP comparison script."""

import numpy as np
from PIL import Image

from greedy_compare import carve_tracked, main
from seamcarve.carve import SEAM_FINDERS, resize
from seamcarve.energy import dual_gradient_energy


def random_image(h=14, w=20, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_carve_tracked_matches_resize():
    img = random_image()
    for key, finder in SEAM_FINDERS.items():
        run = carve_tracked(img, 13, finder, dual_gradient_energy)
        np.testing.assert_array_equal(run["image"], resize(img, 14, 13, finder=finder))
        assert len(run["seams"]) == len(run["costs"]) == 7


def test_tracked_seams_are_in_original_coordinates():
    # Deleting the recorded (original) pixels from the original image must give
    # exactly the carved result.
    img = random_image(seed=1)
    run = carve_tracked(img, 12, SEAM_FINDERS["greedy"], dual_gradient_energy)
    keep = np.ones(img.shape[:2], dtype=bool)
    for seam in run["seams"]:
        assert not np.any(~keep[np.arange(img.shape[0]), seam]), "pixel removed twice"
        keep[np.arange(img.shape[0]), seam] = False
    np.testing.assert_array_equal(img[keep].reshape(14, 12, 3), run["image"])


def test_dp_seams_are_never_more_expensive_on_first_step():
    img = random_image(seed=2)
    dp = carve_tracked(img, 19, SEAM_FINDERS["dp"], dual_gradient_energy)
    gr = carve_tracked(img, 19, SEAM_FINDERS["greedy"], dual_gradient_energy)
    assert dp["costs"][0] <= gr["costs"][0]


def test_main_writes_outputs(tmp_path):
    src = tmp_path / "in.png"
    Image.fromarray(random_image(16, 24)).save(src)
    out = tmp_path / "res"
    assert main([str(src), "--out", str(out), "--width", "18", "--scales", "0.8", "1.0",
                 "--repeats", "1", "--no-charts"]) == 0
    for name in ("seams_first.png", "seams_all.png", "result.png", "greedy_vs_dp.md"):
        assert (out / name).exists(), name
    assert "Greedy" in (out / "greedy_vs_dp.md").read_text(encoding="utf-8")


def test_main_with_all_methods_lists_every_method(tmp_path):
    src = tmp_path / "in.png"
    Image.fromarray(random_image(16, 24, seed=3)).save(src)
    out = tmp_path / "res"
    assert main([str(src), "--out", str(out), "--width", "20", "--scales", "1.0",
                 "--repeats", "1", "--no-charts"]) == 0
    text = (out / "greedy_vs_dp.md").read_text(encoding="utf-8")
    for name in ("Dynamic programming", "Greedy, multi-start", "Greedy, 3-row lookahead"):
        assert name in text


def test_experiments_script(tmp_path):
    from greedy_experiments import main as experiments_main

    src = tmp_path / "img.png"
    Image.fromarray(random_image(12, 18, seed=5)).save(src)
    out = tmp_path / "exp"
    assert experiments_main(["--images", str(src), "--energies", "dual-gradient", "sobel",
                             "--shrink", "0.2", "--out", str(out)]) == 0
    rows = (out / "experiments.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 2 * 4                     # header + 2 energies x 4 methods
    dp_rows = [r for r in rows[1:] if ",dp," in r]
    assert all(",1.0," in r for r in dp_rows)        # DP vs itself = 1.0
    assert (out / "result_img_sobel.png").exists()
