"""Tests for the comparison script."""

import numpy as np
from PIL import Image

from compare import GAP, LABEL_HEIGHT, center_crop, main


def test_center_crop_takes_the_middle():
    arr = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    out = np.asarray(center_crop(Image.fromarray(arr), 4, 2))
    np.testing.assert_array_equal(out, arr[2:4, 2:6])


def test_compare_output_layout(tmp_path):
    rng = np.random.default_rng(0)
    src, dst = tmp_path / "in.png", tmp_path / "cmp.png"
    Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)).save(src)

    assert main([str(src), str(dst), "--width", "10"]) == 0
    # original (16) + three panels of width 10 + 3 gaps; height = 12 + label strip
    assert Image.open(dst).size == (16 + 3 * 10 + 3 * GAP, 12 + LABEL_HEIGHT)


def test_compare_rejects_enlarging(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "cmp.png"
    Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)).save(src)
    assert main([str(src), str(dst), "--width", "9"]) == 1
    assert not dst.exists()
