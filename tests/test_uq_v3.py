"""s4/uq_conformal.py cls3 on synthetic s4/cls_hybrid.py run folders: conformal coverage near nominal, the spread joins and their guards."""
import csv
import json
import shutil

import numpy as np
import pytest

from s4.uq_conformal import crc_fnr, entropy, fnr, main, mondrian_sets, risk_coverage

BD = ["Normal", "Pneumonia Bacterial", "Pneumonia Viral"]


def _run(d, **arrays):
    d.mkdir(parents=True)
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", v)


def _csv(path, key, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key, "spread_lung"])
        w.writerows(rows)


def _z(v):
    return (v - v.mean()) / v.std()


def _nih(root, rng, n=9000, c=6):
    """NIH-like labels and two hybrid seeds (validation 1000-3999, test 5000-8999) -> labels, val idx, test idx, seed-mean pred [n, c]."""
    Y = (rng.random((n, c)) < 0.2).astype(np.float32)
    va, te = np.arange(1000, 4000), np.arange(5000, n)  # unequal sizes: validation and test cannot stand in for each other unnoticed
    preds = []
    for s in range(2):
        p = np.clip(0.3 * Y + 0.7 * rng.random(Y.shape), 0, 1).astype(np.float16)
        _run(root / "nih" / f"nih_hyb_hybrid@densenet121_s{s}", val_pred=p[va], val_idx=va, test_pred=p[te], test_idx=te,
             test_labels=Y[te], test_patient=te)
        preds.append(p.astype(np.float32))
    return Y, va, te, np.mean(preds, 0)


def _bdcxr(root, rng, n=6000):
    """BDCXR-like orig runs, 5 folds x 2 seeds -> labels, image names (prep order), per fold the seed-mean pred and the prep idx."""
    y = rng.integers(0, 3, n)
    w = np.array([1.0, 0.6, 0.3])[y, None]  # weaker true-class signal in classes 1, 2: one marginal quantile covers class 2 at ~0.80
    fold = rng.permutation(n) % 5
    P, I = [], []
    for f in range(5):
        te = np.nonzero(fold == f)[0]
        preds = []
        for s in range(2):
            p = (0.5 * rng.dirichlet([1, 1, 1], len(te)) + 0.5 * np.eye(3)[y[te]] * w[te] * rng.random((len(te), 1))).astype(np.float16)
            _run(root / "bdcxr" / f"bdcxr_hyb_orig@rad-dino_s{s}_f{f}", val_pred=p[:8], val_idx=te[:8], test_pred=p, test_idx=te,
                 test_labels=np.eye(3, dtype=np.float32)[y[te]], test_patient=te)
            preds.append(p.astype(np.float32))
        P.append(np.mean(preds, 0))
        I.append(te)
    return y, [f"{BD[k]}/{BD[k]} ({i}).jpg" for i, k in enumerate(y)], P, I


def test_cls3_nih_crc_near_nominal_and_index_join(tmp_path):
    rng = np.random.default_rng(0)
    Y, va, te, P = _nih(tmp_path, rng)
    p = P[te]
    np.save(tmp_path / "labels.npy", Y)
    spread = {i: rng.random() for i in te[::2]}  # half the test images have a spread, one of them NaN (empty lung mask)
    spread[te[0]] = float("nan")
    rows = list(spread.items())
    _csv(tmp_path / "spread_nih.csv", "index", [rows[k] for k in rng.permutation(len(rows))])
    main(["cls3", "--arch", "densenet121", "--root", str(tmp_path), "--nih_labels", str(tmp_path / "labels.npy"),
          "--spread_nih", str(tmp_path / "spread_nih.csv"), "--out", str(tmp_path / "uq.json")])
    r = json.loads((tmp_path / "uq.json").read_text())
    h = r["nih_hybrid"]
    assert (h["n_runs"], h["n_cal"], h["n_test"]) == (2, 3000, 4000)
    # seed mean of val_pred.npy with labels.npy at val_idx (here 0.543; test predictions 0.547, one seed 0.632)
    assert h["lam"] == crc_fnr(P[va], Y[va], 0.1)
    # evaluated on the test seed means (0.105; the validation FNR at this lam is 0.099), near nominal
    assert h["fnr_test"] == pytest.approx(fnr(p, Y[te], h["lam"]), rel=1e-12) and 0.08 <= h["fnr_test"] <= 0.12
    assert h["mean_set_size"] == pytest.approx((p >= 1 - h["lam"]).sum(1).mean(), rel=1e-12)
    assert {"nih_orig", "nih_ctrl", "bdcxr_hybrid"} <= set(r["missing"])
    ent, loss = entropy(p, True), np.nanmean((p - Y[te]) ** 2, 1)
    rc, aurc = risk_coverage(-ent, loss)  # score A, all test images
    assert np.allclose(h["risk_coverage_brier"], rc) and h["aurc_brier"] == pytest.approx(aurc, rel=1e-9)
    s = np.array([spread.get(i, np.nan) for i in te])  # the join on the prep index, rebuilt here
    ok = np.isfinite(s)
    b = h["score_b"]
    assert b["n"] == ok.sum() == len(te) // 2 - 1
    assert b["aurc"] == pytest.approx(risk_coverage(-(_z(ent[ok]) + _z(s[ok])), loss[ok])[1], rel=1e-9)
    assert b["aurc_score_a_same_images"] == pytest.approx(risk_coverage(-ent[ok], loss[ok])[1], rel=1e-9)


def test_cls3_bdcxr_cross_conformal_mondrian_and_name_join(tmp_path):
    rng = np.random.default_rng(1)
    y, ids, Ps, Is = _bdcxr(tmp_path, rng)
    (tmp_path / "ids.json").write_text(json.dumps(ids))
    spread = {nm: rng.random() for nm in rng.permutation(ids)[: len(ids) // 2]}
    _csv(tmp_path / "spread_bdcxr.csv", "image", list(spread.items()))
    main(["cls3", "--arch", "rad-dino", "--root", str(tmp_path), "--bdcxr_ids", str(tmp_path / "ids.json"),
          "--spread_bdcxr", str(tmp_path / "spread_bdcxr.csv"), "--out", str(tmp_path / "uq.json")])
    r = json.loads((tmp_path / "uq.json").read_text())["bdcxr_orig"]
    assert (r["n"], r["n_runs"], r["folds"]) == (len(y), 10, ["f0", "f1", "f2", "f3", "f4"])
    # cross-conformal: fold f calibrated on the other four folds' seed means only (with its own fold: other coverage and set size)
    S = np.concatenate([mondrian_sets(np.concatenate(Ps[:f] + Ps[f + 1:]), y[np.concatenate(Is[:f] + Is[f + 1:])], Ps[f], 0.1)
                        for f in range(5)])
    P, I = np.concatenate(Ps), np.concatenate(Is)
    yi = y[I]
    assert r["coverage"] == pytest.approx(S[np.arange(len(yi)), yi].mean(), rel=1e-12)
    assert r["coverage_per_class"] == pytest.approx([S[yi == k, k].mean() for k in range(3)], rel=1e-12)
    assert r["mean_set_size"] == pytest.approx(S.sum(1).mean(), rel=1e-12)
    assert 0.87 <= r["coverage"] <= 0.93 and min(r["coverage_per_class"]) >= 0.87  # Mondrian: the weak class 2 is covered too
    ent, err = entropy(P, False), (P.argmax(1) != yi).astype(float)
    rc, aurc = risk_coverage(-ent, err)  # score A, all images
    assert np.allclose(r["risk_coverage"], rc) and r["aurc"] == pytest.approx(aurc, rel=1e-9)
    s = np.array([spread.get(ids[i], np.nan) for i in I])  # the join on the image name, rebuilt here
    ok = np.isfinite(s)
    assert r["score_b"]["n"] == ok.sum() == len(y) // 2
    assert r["score_b"]["aurc"] == pytest.approx(risk_coverage(-(_z(ent[ok]) + _z(s[ok])), err[ok])[1], rel=1e-9)


def test_cls3_join_unavailable_and_mismatch_guards(tmp_path):
    rng = np.random.default_rng(2)
    Y, _, _, _ = _nih(tmp_path, rng)
    _, ids, _, _ = _bdcxr(tmp_path, rng, n=1500)
    shutil.copytree(tmp_path / "bdcxr" / "bdcxr_hyb_orig@rad-dino_s0_f3", tmp_path / "bdcxr" / "bdcxr_hyb_soft@rad-dino_s0_f3")
    _csv(tmp_path / "s.csv", "image", [(ids[0], 0.1)])
    out = ["--out", str(tmp_path / "o.json")]
    base = ["cls3", "--arch", "rad-dino", "--root", str(tmp_path)] + out
    args = base + ["--spread_bdcxr", str(tmp_path / "s.csv")]
    main(args + ["--bdcxr_ids", str(tmp_path / "absent.json")])
    r = json.loads((tmp_path / "o.json").read_text())
    assert r["bdcxr_orig"]["score_b"].startswith("unavailable") and "nih_hybrid" in r["missing"]  # the NIH runs are densenet121
    assert "bdcxr_soft (only fold f3)" in r["missing"] and "bdcxr_soft" not in r  # one fold cannot be calibrated on the others
    (tmp_path / "ids.json").write_text(json.dumps(ids[1:] + ids[:1]))  # names out of prep order
    with pytest.raises(AssertionError, match="does not match"):
        main(args + ["--bdcxr_ids", str(tmp_path / "ids.json")])
    main(base + ["--bdcxr_ids", str(tmp_path / "ids.json")])  # no spread CSV: the names are not read, the sets do not need them
    assert "score_b" not in json.loads((tmp_path / "o.json").read_text())["bdcxr_orig"]
    Y[-1] = 1 - Y[-1]  # local labels differ from the runs' test_labels.npy on one test image
    np.save(tmp_path / "labels.npy", Y)
    with pytest.raises(AssertionError, match="does not match"):
        main(["cls3", "--arch", "densenet121", "--root", str(tmp_path), "--nih_labels", str(tmp_path / "labels.npy")] + out)
