import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_release as R  # noqa: E402


@pytest.mark.parametrize("rel, why", [
    ("Kaggle_Token.txt", "credentials"), ("context/HF_Token.txt.txt", "credentials"), ("x/kaggle.json", "credentials"),
    ("outputs/v4/ct-a/results/ct_xm_ct_s0/model.pt", "data/weights"), ("results/v4/down/nih/r_s0/test_pred.npy", "data/weights"),
    ("results/final/audit_ribcxr/w16_s0.csv", "restricted"), ("results/final/rib/swin.csv", "restricted"),
    ("outputs/prep_dl/ac_9-s4-prep-vin/ann.json", "restricted"), ("results/bdcxr/audit_bdcxr/ours.csv", "restricted"),
    ("results/final/node21/ann.json", "restricted"), ("outputs/ctprep/ct/status.json", "restricted"),
    ("results/final/audit_ribcxr/stats_held.json", None), ("outputs/v4/ct-a/results/ct_xm_ct_s0/result.json", None),
    ("results/v3/down/vin/vin_hyb_orig@densenet121_s0/log.jsonl", None), ("results/final/szch/w16_s0.csv", None),
    ("s4/xm.py", None), ("results/v4/pointing/nih_hyb_orig@densenet121_s0.json", None), ("tools/convnext_screen.py", None),
    ("results/v3/xai/nih_hyb_hybrid@densenet121_s0.json", None), ("results/v4/xai/ct_xmx_xraykd_s0/result.json", None),
    ("results/v4/down/ct/ct_xm_ct_s5/log.jsonl", None),
    # code, plans and manuscript tables that only name a restricted set are published
    ("protocol/SAP_bdcxr.md", None), ("tools/bdcxr_stats.py", None), ("s4/eval_ribcxr.py", None), ("submission/tables/bdcxr_cls.tex", None),
    ("kernels/prep_ac_1_vin/run_prep.py", None), ("kernels/down_ac_1_s4-v4-ct-a/kernel-metadata.json", None),
    ("paper/main_cls.tex", "superseded"), ("paper/manuscript_lesion_audit.docx", "superseded"), ("paper/REFS_TODO.md", "superseded"),
    ("tools/docx/build.py", "superseded"), ("paper/sections/data.tex", None), ("paper/refs.bib", None),
    ("outputs/smoke_qp/result.json", "scratch"), ("outputs/dry2/log.jsonl", "scratch"), ("outputs/gt_ac_4/s4-gputest.log", "scratch"),
    ("outputs/queue_wave4.state.json", "scratch"), ("outputs/launch_smoke5c.log", "scratch"), ("outputs/szch_pairs.png", "scratch"),
    ("outputs/queue_v3_r1.json", "scratch"), ("outputs/.par_lines_before", "scratch"), ("outputs/watch_bb.sh", "scratch"),
    ("kernels/gputest_ac_4/run.py", "scratch"), ("kernels/down_ac_5_s4-dn-smoke5/run_down.py", "scratch"),
    ("outputs/screen_r4.csv", None), ("outputs/w1_dl/s4-w1-1/log.jsonl", None), ("outputs/audit_final/fin.log", None),
    ("outputs/pilot_lr/result.json", None), ("kernels/down_ac_9_s4-v4-dn121/run_down.py", None),
    ("s4/data/xray_tables.json", None), ("s4/data/ct_subset.json", "restricted"), ("s4/data/ct_skip.txt", "restricted"),
])
def test_rules(rel, why):
    assert R.rule(rel) == why


def test_secret_patterns_and_clean_text():
    assert R.SECRET.search("token = 'hf_" + "a" * 34 + "'") and R.SECRET.search('{"username": "u", "key": "' + "0" * 32 + '"}')
    assert not R.SECRET.search("KAGGLE_API_TOKEN=token(label)  # read from the token file, never printed")


def test_build_redacts_usernames_and_drops_scratch(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "kernels" / "train_ac_2_x").mkdir(parents=True)
    (root / "outputs" / "smoke_run").mkdir(parents=True)
    (root / "outputs" / "w1_dl" / "alice_k").mkdir(parents=True)
    (root / "config.yaml").write_text("accounts:\n  ac_1: alice\n  ac_2: bobby   # replaced carol (no phone)\nlicensed_only_on: ac_1\n")
    (root / "kernels" / "train_ac_2_x" / "kernel-metadata.json").write_text('{"id": "bobby/s4-x", "by": "carol"}')
    (root / "outputs" / "smoke_run" / "result.json").write_text("{}")
    (root / "outputs" / "w1_dl" / "alice_k" / "result.json").write_text('{"run": "/kaggle/input/alice/x"}')
    (root / "overleaf_submission_preview.pdf").write_bytes(b"%PDF")
    for d in ("data", "s4/data"):  # raw data at the top is pruned; the package's own tables are not
        (root / d).mkdir(parents=True)
        (root / d / "tables.json").write_text("{}")
    monkeypatch.setattr(R, "ROOT", root)
    out = tmp_path / "snap"
    kept, dropped = R.build(out, [])
    assert (out / "kernels/train_ac_2_x/kernel-metadata.json").read_text() == '{"id": "ac_2/s4-x", "by": "ac_2_former"}'
    assert (out / "outputs/w1_dl/ac_1_k/result.json").read_text() == '{"run": "/kaggle/input/ac_1/x"}'
    assert (out / "config.yaml").read_text().startswith("accounts:\n  ac_1: ac_1\n  ac_2: ac_2")
    assert (out / "manuscripts/manuscript.pdf").is_file() and not (out / "outputs/smoke_run").exists()
    assert (out / "s4/data/tables.json").is_file() and not (out / "data").exists()
    assert dropped == {"scratch": 1} and (out / "EXCLUDED.md").is_file()
    with pytest.raises(SystemExit):  # the folder is no longer empty
        R.build(out, [])
    (root / "figure.png").write_bytes(b"\x89PNG alice")  # a username that cannot be redacted
    with pytest.raises(SystemExit, match="username"):
        R.build(tmp_path / "snap2", [])
    assert not (tmp_path / "snap2").exists()
