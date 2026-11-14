import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pointing_v4 as P  # noqa: E402


def test_seed_mean_rates_split_and_zero_maps(tmp_path):
    rows = lambda h: [dict(image=i, cls=c, hits=dict(cam_0=x, ig_S=False)) for (i, c), x in zip([(1, "Mass"), (2, "Effusion"), (3, "Nodule"), (3, "Effusion")], h)]
    for s, h in enumerate([(True, False, True, True), (True, True, False, False)]):
        (tmp_path / f"r_s{s}.json").write_text(json.dumps(dict(rows=rows(h), hit_rate_cam_0=sum(h) / 4)))
    r = P.summarize(sorted(tmp_path.glob("*.json")), n=200)
    assert r["n_targets"] == 4 and r["n_focal"] == 2 and r["n_images"] == 3 and list(r["maps"]) == ["cam_0"]
    m = r["maps"]["cam_0"]
    assert m["all"] == 0.625 and m["focal"] == 0.75 and m["other"] == 0.5 and m["lo"] <= m["all"] <= m["hi"]
    assert r["per_seed"]["r_s0"] == {"cam_0": 0.75}
