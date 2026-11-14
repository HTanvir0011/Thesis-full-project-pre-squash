import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import collect_v4 as C  # noqa: E402


def run(src, slug, tag, res, extra=b"x"):
    d = src / slug / "results" / tag
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(res))
    (d / "pred.npy").write_bytes(extra)
    (d / "model.pt").write_bytes(b"w")


def test_layout_duplicates_conflicts_and_skips(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    run(src, "p1-b", "qp_xm_xraykd_s3", {"auc": 0.79})
    run(src, "p1-b-r", "qp_xm_xraykd_s3", {"auc": 0.79})  # --prev copy: same result -> copied once
    run(src, "p1-b-r", "nih_xm_transfer-xraykd_s3", {"auc": 0.6})
    run(src, "xmx", "ct_xmx_xraykd_s0", {"r": 0.1})
    run(src, "dn121", "nih_xmh_dn121_s0", {"n": 1})
    run(src, "ct-a", "ct_xm_ct_s2", {"auc": 0.55})
    run(src, "a", "qp_xm_xray_s0", {"auc": 0.80})
    run(src, "b", "qp_xm_xray_s0", {"auc": 0.81})
    run(src, "c", "vin_xm_transfer-xray_s0", {"auc": 0.7}, extra=b"")
    rep = C.collect(src, dst)
    assert (dst / "down/qp/qp_xm_xraykd_s3/result.json").exists() and not (dst / "down/qp/qp_xm_xraykd_s3/model.pt").exists()
    assert (dst / "down/nih/nih_xm_transfer-xraykd_s3").is_dir() and (dst / "xai/ct_xmx_xraykd_s0").is_dir()
    assert (dst / "halluc/nih_xmh_dn121_s0").is_dir() and not (dst / "down/ct").exists()
    assert rep["duplicates"] == {"qp_xm_xraykd_s3": ["p1-b", "p1-b-r"]} and rep["skipped"] == ["ct_xm_ct_s2"]
    assert list(rep["conflicts"]) == ["qp_xm_xray_s0"] and not (dst / "down/qp/qp_xm_xray_s0").exists()
    assert rep["incomplete"] == ["vin_xm_transfer-xray_s0"]
    again = C.collect(src, dst)
    assert not again["copied"] and len(again["kept"]) == 4
