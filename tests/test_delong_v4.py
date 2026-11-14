import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import delong_v4 as D  # noqa: E402


def test_family_drops_nan_labels_and_holm_adjusts():
    rng = np.random.default_rng(0)
    y = (rng.random((400, 2)) < 0.3).astype(float)
    y[:50, 1] = np.nan
    good, poor = y + rng.normal(0, 0.5, y.shape), y + rng.normal(0, 3.0, y.shape)
    r = D.family(y, {"a": np.nan_to_num(good), "b": np.nan_to_num(poor)}, [("a", "b")], ["c0", "c1"])
    assert list(r) == ["a-b:c0", "a-b:c1"] and all(v["diff"] > 0 and v["p_holm"] >= v["p"] for v in r.values())
