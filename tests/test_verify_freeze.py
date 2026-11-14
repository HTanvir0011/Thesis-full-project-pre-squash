import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_freeze as V  # noqa: E402


def test_statuses_and_exit_code(tmp_path, capsys):
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    (tmp_path / "protocol").mkdir()
    (tmp_path / "s4").mkdir()
    (tmp_path / "protocol" / "SAP_v4.md").write_bytes(b"plan")
    (tmp_path / "s4" / "xm.py").write_bytes(b"edited later")
    (tmp_path / "config.yaml").write_bytes(b"cfg")
    freeze = {"created": "2026-11-08", "config_sha256": sha(b"cfg"), "primary": "H11",
              "files": {"SAP_v4.md": sha(b"plan"), "s4/xm.py": sha(b"as frozen")},
              "splits": {"ac_6-s4-prep-nih/nih/split.npy": sha(b"split")}, "models": {"arch": "swinv2_tiny_window8_256.ms_in1k"}}
    (tmp_path / "protocol" / "freeze_v4.json").write_text(json.dumps(freeze))
    assert V.main(tmp_path) == 0
    out = capsys.readouterr().out
    assert "freeze_v4.json: 2 OK, 1 CHANGED, 1 absent" in out
    assert "  CHANGED s4/xm.py" in out and "split.npy" not in out  # absent files are only counted
    (tmp_path / "protocol" / "SAP_v4.md").write_bytes(b"plan, edited")
    assert V.main(tmp_path) == 1


def test_flat_keys_and_model_lists():
    d = {"sap_v2_sha256": "a" * 64, "sap_v1_sha256": "b" * 64, "models": {"ours": [{"file": "w/final_ema.pt", "sha256": "c" * 64}]},
         "szch_dev": ["197.png"]}
    assert sorted(V.recorded(d)) == [("SAP.md", "b" * 64), ("SAP_v2.md", "a" * 64), ("w/final_ema.pt", "c" * 64)]
