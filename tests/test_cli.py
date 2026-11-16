"""End-to-end tests for the command-line resizer."""

import numpy as np
from PIL import Image

from cli import main


def write_image(path, h=12, w=16, mode="RGB"):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).convert(mode).save(path)


def test_cli_resizes_width_and_height(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out" / "result.png"
    write_image(src)
    assert main([str(src), str(dst), "--width", "10", "--height", "9"]) == 0
    assert Image.open(dst).size == (10, 9)          # Pillow reports (width, height)


def test_cli_keeps_unspecified_dimension(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    write_image(src)
    assert main([str(src), str(dst), "--width", "11"]) == 0
    assert Image.open(dst).size == (11, 12)


def test_cli_accepts_grayscale_and_rgba_files(tmp_path):
    for mode in ("L", "RGBA"):
        src, dst = tmp_path / f"in_{mode}.png", tmp_path / f"out_{mode}.png"
        write_image(src, mode=mode)
        assert main([str(src), str(dst), "--height", "8"]) == 0
        assert Image.open(dst).size == (16, 8)


def test_cli_rejects_enlarging(tmp_path, capsys):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    write_image(src)
    assert main([str(src), str(dst), "--width", "20"]) == 1
    assert "Error" in capsys.readouterr().err
    assert not dst.exists()


def test_cli_sobel_energy(tmp_path):
    src, dst = tmp_path / "in.png", tmp_path / "out.png"
    write_image(src)
    assert main([str(src), str(dst), "--width", "12", "--energy", "sobel"]) == 0
    assert Image.open(dst).size == (12, 12)


def test_cli_fast_gives_identical_output(tmp_path):
    src = tmp_path / "in.png"
    write_image(src)
    slow, fast = tmp_path / "slow.png", tmp_path / "fast.png"
    assert main([str(src), str(slow), "--width", "10", "--height", "9"]) == 0
    assert main([str(src), str(fast), "--width", "10", "--height", "9", "--fast"]) == 0
    np.testing.assert_array_equal(np.asarray(Image.open(fast)), np.asarray(Image.open(slow)))
