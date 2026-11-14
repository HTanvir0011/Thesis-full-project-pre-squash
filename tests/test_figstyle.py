import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import figstyle as fs  # noqa: E402


def test_saved_at_print_size(tmp_path):
    fig, ax = fs.figure("single", aspect=0.75)
    ax.plot([0, 1], [0, 1], label="x")
    ax.set_xlabel("x")
    ax.legend()
    fs.save(fig, str(tmp_path / "f"))
    w, h = Image.open(tmp_path / "f.png").size
    assert abs(w - 88 / 25.4 * 600) <= 2 and abs(h - 66 / 25.4 * 600) <= 2  # 88 x 66 mm at 600 dpi, not trimmed
    assert (tmp_path / "f.pdf").stat().st_size > 0


def test_small_text_and_odd_width_are_refused(tmp_path):
    fig, ax = fs.figure("double", height_mm=60)
    ax.text(0.5, 0.5, "tiny", fontsize=6)
    with pytest.raises(ValueError, match="6.0 pt"):
        fs.save(fig, str(tmp_path / "g"))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 3))
    with pytest.raises(ValueError, match="not a standard column"):
        fs.save(fig, str(tmp_path / "h"))
    with pytest.raises(ValueError, match="page limit"):
        fs.figure("single", height_mm=300)
