"""s4/spread.py NIH mode (test-split listing, index column, missing files, per-GPU parts and their merge), the kernel template and the
ops/launch_spread.py arguments."""
import csv
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import s4.spread as S

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ["train", "test", "test", "val", "test", "test", "test"]


def img(i):
    return np.random.default_rng(i).integers(0, 256, (16, 16), dtype=np.uint8)


def fake_nih(tmp_path, split):
    """Prep folder nih/ (ids.json, split.npy) and a raw root with image img(i) per index over two images_*/images folders; index 4 has no
    file and index 5 an unreadable one."""
    names = [f"{i:08d}_000.png" for i in range(len(split))]
    prep = tmp_path / "prep" / "nih"
    prep.mkdir(parents=True)
    (prep / "ids.json").write_text(json.dumps(names))
    np.save(prep / "split.npy", np.array(split))
    for i, n in enumerate(names):
        d = tmp_path / "raw" / f"images_00{1 + i % 2}" / "images"
        d.mkdir(parents=True, exist_ok=True)
        if i == 5:
            (d / n).write_bytes(b"not a png")
        elif i != 4:
            cv2.imwrite(str(d / n), img(i))
    return tmp_path / "raw", prep, names


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_nih_lists_test_split_in_index_order(tmp_path, capsys):
    root, prep, names = fake_nih(tmp_path, SPLIT)
    got = list(S.images("nih", root, prep))
    assert [(n, e) for n, _, e in got] == [(names[i], {"index": i}) for i in (1, 2, 6)]  # test split only, prep index alongside
    assert all(np.array_equal(im, img(e["index"])) for _, im, e in got)  # each name's own image
    out = capsys.readouterr().out
    assert f"missing {names[4]}" in out and f"unreadable {names[5]}" in out  # skipped with a log line, not a crash
    assert [e["index"] for _, _, e in S.images("nih", root, prep, limit=2)] == [1, 2]  # smoke test: the first N test images
    with pytest.raises(AssertionError, match="no images"):  # a wrong --root fails at once
        list(S.images("nih", tmp_path / "prep", prep))


def test_workers_get_whole_batches(tmp_path):
    root, prep, _ = fake_nih(tmp_path, ["val"] * 6 + ["test"] * 70)  # test indices 6..75: batches 6-37, 38-69, 70-75

    def part(g, **kw):
        return [e["index"] for _, _, e in S.images("nih", root, prep, worker=g, nworkers=2, **kw)]
    assert [part(0), part(1)] == [list(range(6, 38)) + list(range(70, 76)), list(range(38, 70))]
    assert [part(0, limit=40), part(1, limit=40)] == [list(range(6, 38)), list(range(38, 46))]  # the limit applies before the split


def test_merge_orders_parts_by_index(tmp_path):
    def row(i):
        return dict(dataset="nih", image=f"{i}.png", lung_frac="0.4", spread_lung="nan" if i == 9 else "0.0123456789012345",
                    spread_all="0.02", index=str(i))
    S.write([row(10), row(9)], tmp_path / "p0.csv")
    S.write([row(2)], tmp_path / "p1.csv")
    assert S.merge([tmp_path / "p0.csv", tmp_path / "p1.csv"], tmp_path / "m.csv") == 3
    assert read(tmp_path / "m.csv") == [row(2), row(9), row(10)]  # numeric index order, values as written
    with pytest.raises(AssertionError):
        S.merge([tmp_path / "p0.csv", tmp_path / "p0.csv"], tmp_path / "m2.csv")  # an image scored by two workers


def test_cli_rejects_misplaced_options(monkeypatch):
    monkeypatch.setattr(S, "load_suppressor", lambda *a: pytest.fail("options were not checked before the models load"))
    for extra in (["--dataset", "nih"], ["--dataset", "bdcxr", "--prep", "p"], ["--dataset", "szch", "--limit", "5"],
                  ["--dataset", "bdcxr", "--nworkers", "2"], ["--dataset", "nih", "--prep", "p", "--worker", "2", "--nworkers", "2"]):
        monkeypatch.setattr(sys, "argv", ["spread", "--weights", "a.pt+b.pt", "--root", "r", "--out", "o.csv", *extra])
        with pytest.raises(SystemExit):
            S.main()


def test_main_nih_rows_and_parts_equal_one_process(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    root, prep, names = fake_nih(tmp_path, ["val"] * 6 + ["test"] * 70)
    (tmp_path / "bdcxr" / "Normal").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "bdcxr" / "Normal" / "Normal (1).jpg"), img(0))
    torch.manual_seed(0)
    members = [torch.nn.Conv2d(1, 1, 3, padding=1).cuda().eval() for _ in range(3)]  # random members disagree
    monkeypatch.setattr(S, "load_suppressor", lambda w, dev: types.SimpleNamespace(members=members))
    monkeypatch.setattr(S, "lung_masks", lambda xs, dev: [np.ones((1024, 1024), bool) for _ in xs])

    def run(out, *args):
        monkeypatch.setattr(sys, "argv", ["spread", "--weights", "a+b+c", *args, "--out", str(tmp_path / out)])
        S.main()
    nih = ["--dataset", "nih", "--root", str(root), "--prep", str(prep)]
    run("one.csv", *nih)
    for g in (0, 1):
        run(f"part{g}.csv", *nih, "--worker", str(g), "--nworkers", "2")
    run("bdcxr.csv", "--dataset", "bdcxr", "--root", str(tmp_path / "bdcxr"))
    one, bd = read(tmp_path / "one.csv"), read(tmp_path / "bdcxr.csv")
    assert list(one[0]) == ["dataset", "image", "lung_frac", "spread_lung", "spread_all", "index"]
    assert [(r["image"], r["index"]) for r in one] == [(names[i], str(i)) for i in range(6, 76)]
    assert all(float(r["lung_frac"]) == 1 and float(r["spread_all"]) > 0 for r in one)
    assert S.merge([tmp_path / "part0.csv", tmp_path / "part1.csv"], tmp_path / "merged.csv") == 70
    assert (tmp_path / "merged.csv").read_bytes() == (tmp_path / "one.csv").read_bytes()  # two GPUs give the one-process file
    assert list(bd[0]) == ["dataset", "image", "lung_frac", "spread_lung", "spread_all"] and bd[0]["image"] == "Normal/Normal (1).jpg"


def test_launch_spread_arguments():
    sys.path.insert(0, str(ROOT / "ops"))
    from launch_spread import parse
    w = "cnx_t16_s0.pt+cnx_t16_s1.pt+cnx_t16_s2.pt"
    assert parse(["ac_6", "k", "--weights", "s4-wts3", w]) == ("ac_6", "k", "s4-wts3", w, "s4-prep-nih")
    assert parse(["ac_6", "k", "--weights", "s4-wts3", "--files", w, "--prep", "s4-prep-nih"]) == ("ac_6", "k", "s4-wts3", w, "s4-prep-nih")
    e = "+".join(f"v2__qp_twin16_s{s}_cnx2_300k.pt" for s in range(3))
    assert parse(["ac_7", "k", "--weights", "s4-wte", e, "--prep", "p"])[2:] == ("s4-wte", e, "p")
    for bad in (["ac_6", "k"], ["ac_6", "k", "--weights", "s4-wts3"], ["ac_6", "k", "--weights", "s4-wts3", "a.pt"],
                ["ac_6", "k", "--weights", "s4-wts3", "a.pt+b.pt"],  # not the frozen 3-member ensemble
                ["ac_6", "k", "--weights", "s4-wts3", w, "--prp", "p"], ["ac_6", "k", "--weights", "s4-wts3", w, "--prep"],
                ["ac_6", "k", "--weights", "s4-wts3", w, "--prep", "--files"], ["ac_6", "k", "--weights", "s4-wts3", w, "--prep", "u/p"],
                ["ac_6", "k", "--weight", "s4-wts3", w], ["ac_6", "k", "--weights", "u/s4-wts3", w],
                ["ac_6", "k", "--weights", "--files", w], ["ac_6", "k", "--weights", "s4-wts3", "a.pt+b.pt+c.npy"],
                ["ac_6", "k", "--weights", "s4-wts3", w, "--prep", "p", "--limit", "5"]):
        with pytest.raises(SystemExit):  # a typo is an error, not silently ignored
            parse(bad)


def test_template_compiles_after_substitution():
    compile((ROOT / "kernels/spread/run_spread.py").read_text().replace("__WEIGHTS__", repr("a.pt+b.pt+c.pt")), "run_spread.py", "exec")
