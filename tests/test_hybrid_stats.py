import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hybrid_stats import holm_families, pointing  # noqa: E402
from s4.stats import holm  # noqa: E402


def test_holm_families_span_both_encoders():
    """H6: one family over encoders x data sets; H7: one family over encoders x data sets x {ctrl, early} (DEVIATIONS.md 2026-11-04)."""
    ps = iter(np.linspace(0.001, 0.3, 18))
    enc = {a: {ds: {h: {"auroc": {"p": float(next(ps))}} for h in ("H6", "H7_ctrl", "H7_early")} for ds in ("nih", "vin", "bdcxr")}
           for a in ("densenet121", "rad-dino")}
    raw6 = [enc[a][ds]["H6"]["auroc"]["p"] for a in enc for ds in ("nih", "vin", "bdcxr")]
    raw7 = [enc[a][ds][h]["auroc"]["p"] for a in enc for ds in ("nih", "vin", "bdcxr") for h in ("H7_ctrl", "H7_early")]
    holm_families(enc)
    assert [enc[a][ds]["H6"]["auroc"]["p_holm"] for a in enc for ds in ("nih", "vin", "bdcxr")] == holm(raw6)
    assert [enc[a][ds][h]["auroc"]["p_holm"] for a in enc for ds in ("nih", "vin", "bdcxr") for h in ("H7_ctrl", "H7_early")] == holm(raw7)


def _xai(path, rows):
    path.write_text(json.dumps(dict(rows=rows)))
    return path


def test_pointing_interaction(tmp_path):
    """Focal targets: the soft-tissue stream hits, the original does not; other targets: neither -> interaction 1 with a tight interval."""
    rows = [dict(image=i, cls="Nodule" if i < 20 else "Effusion", hits=dict(cam_0=False, cam_1=i < 20, cam_2=False)) for i in range(40)]
    files = [_xai(tmp_path / f"nih_hyb_hybrid@densenet121_s{s}.json", rows) for s in range(2)]
    r = pointing(files, n=500)
    assert r["n_seeds"] == 2 and r["n_targets"] == 40 and r["n_focal"] == 20
    assert r["interaction"] == 1.0 and r["lo"] == 1.0 and r["p"] < 0.01
    assert r["hit_rates"]["cam_1"] == dict(focal=1.0, other=0.0)


def test_pointing_null_covers_zero(tmp_path):
    rng = np.random.default_rng(0)
    files = []
    for s in range(3):
        rows = [dict(image=i, cls="Mass" if i % 2 else "Cardiomegaly", hits=dict(cam_0=bool(rng.random() < .4), cam_1=bool(rng.random() < .4)))
                for i in range(300)]
        files.append(_xai(tmp_path / f"s{s}.json", rows))
    r = pointing(files, n=1000)
    assert r["lo"] < 0 < r["hi"] and r["p"] > 0.05


def test_tables_from_saved_statistics(tmp_path):
    import hybrid_stats as H
    c = dict(diff=0.001, lo=-0.002, hi=0.004, p=0.4, p_holm=0.8)
    e = dict(n=10, macro_auroc={k: 0.8 for k in H.CONFIGS}, macro_auprc={k: 0.3 for k in H.CONFIGS}, H6=dict(auroc=c, auprc=c),
             per_class_delong={f"hybrid-{r}:{i}": dict(auc1=0.81, auc2=0.8, **c) for r in ("orig", "early") for i in (4, 5)})
    out = dict(archs=["densenet121"], encoders={"densenet121": {"nih": e}},
               H10={"densenet121": dict(n_targets=5, n_focal=2, interaction=0.1, lo=0.05, hi=0.15, p=0.01,
                                        hit_rates={"cam_0": dict(focal=0.2, other=0.4), "cam_1": dict(focal=0.4, other=0.5)})})
    f = tmp_path / "s.json"
    f.write_text(json.dumps(out))
    H.main(["--tables_from", str(f), "--tables", str(tmp_path / "t")])
    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == ["v3_macro.tex", "v3_per_class.tex", "v3_pointing.tex", "v3_primary.tex"]
    assert "Nodule" in (tmp_path / "t/v3_per_class.tex").read_text()
    assert "+0.200 & +0.100 & +0.100 [+0.050, +0.150], p = 0.01" in (tmp_path / "t/v3_pointing.tex").read_text()
