"""Smoke tests for the benchmark script (tiny sizes so it runs in a second)."""

import csv

import numpy as np
import pytest
from PIL import Image

from benchmark import COLUMNS, linear_fit_r2, main


def test_linear_fit_detects_perfect_line():
    a, r2 = linear_fit_r2([1, 2, 3, 4], [3, 5, 7, 9])
    assert a == pytest.approx(2)
    assert r2 == pytest.approx(1)


def test_benchmark_writes_all_outputs(tmp_path):
    src = tmp_path / "in.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)).save(src)
    out = tmp_path / "bench"

    assert main(["--image", str(src), "--out", str(out), "--seams", "3",
                 "--repeats", "1", "--scales", "0.5", "1.0"]) == 0

    with (out / "benchmark.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert list(rows[0]) == COLUMNS
    assert int(rows[1]["pixels"]) == 40 * 60
    assert "R²" in (out / "benchmark.md").read_text(encoding="utf-8")
    assert (out / "time_per_seam.png").exists()
    assert (out / "dp_memory.png").exists()
