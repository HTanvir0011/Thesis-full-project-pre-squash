"""v4 tables for the thesis (paper/tables/v4_*.tex) from the saved statistics; nothing is recomputed.

python tools/xm_tables.py [--stats results/final_xm_stats.json] [--v4 results/v4] [--ct_runs outputs/v4] [--tables paper/tables]
Reads tools/xm_stats.py's output, <v4>/delong.json (tools/delong_v4.py), <v4>/pointing_summary.json (tools/pointing_v4.py),
<v4>/uq_cls_<model>.json (s4/uq_conformal.py cls3), the per-seed result.json of the collected runs (<v4>/down) and, for the CT teacher
seeds, every CT teacher run of <ct_runs>/ct-*/results (the collapsed and the unused seed included; skipped when that folder is absent).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from final_stats import tex  # noqa: E402
from hybrid_stats import fmt  # noqa: E402
from s4.prep import NIH  # noqa: E402
from s4.xm_labels import EXACT_NIH, SHARED  # noqa: E402

MODELS = {"F_in": "F$_\\mathrm{in}$", "F_p0": "F$_\\mathrm{p0}$", "F_p1": "F$_\\mathrm{p1}$", "DN121": "DenseNet-121 (v3)"}
FINDING = {"lung_opacity": "lung opacity", "pleural_effusion": "pleural effusion", "atelectasis": "atelectasis",
           "cardiomegaly": "cardiomegaly", "consolidation": "consolidation", "lung_lesion": "lung lesion"}
MAPS = {"cam_0": "Grad-CAM++", "attn": "decoder attention", "ig_X": "Integrated Gradients"}
CT_SLOT = {0: "slot 0", 1: "slot 1", 2: "collapsed, not used", 3: "slot 3", 4: "slot 4", 5: "slot 2 (replaces seed 2)", 6: "unused spare"}


def p3(p):
    return f"{p:.3g}"


def seeds(d, pattern, key):
    """{seed: metric} of the runs d/pattern (result.json key)."""
    return {int(f.parent.name.rsplit("_s", 1)[1]): json.loads(f.read_text())[key] for f in sorted(d.glob(f"{pattern}/result.json"))}


def write_tables(s, v4, ct_runs, out):
    out.mkdir(parents=True, exist_ok=True)
    cm, cm0 = s["crossmodal_xraykd"], s["crossmodal_xray"]
    rows = [["H11 (primary)", r"F$_\mathrm{p1}$ $-$ F$_\mathrm{in}$, mean AUROC of the 5 shared findings", fmt(s["H11"]), p3(s["H11"]["p"]),
             p3(s["H11"]["p_holm"])],
            ["H12", r"F$_\mathrm{p1}$ $-$ F$_\mathrm{p0}$, same", fmt(s["H12_p1_p0"]), p3(s["H12_p1_p0"]["p"]), p3(s["H12_p1_p0"]["p_holm"])],
            ["H12", r"F$_\mathrm{p0}$ $-$ F$_\mathrm{in}$, same", fmt(s["H12_p0_in"]), p3(s["H12_p0_in"]["p"]), p3(s["H12_p0_in"]["p_holm"])],
            ["H13", r"F$_\mathrm{in}$ $-$ DenseNet-121, NIH macro-AUROC (14 classes)", fmt(s["H13"]), p3(s["H13"]["p"]), "--"],
            ["H14", "P1: matched $-$ mismatched CT/X-ray map correlation", fmt(cm["H14"]), p3(cm["H14"]["p"]), "--"],
            ["(descriptive)", "P0: matched $-$ mismatched CT/X-ray map correlation", fmt(cm0["H14"]), p3(cm0["H14"]["p"]), "--"],
            ["H15", r"F$_\mathrm{p1}$ $-$ F$_\mathrm{in}$, rib-emphasis flip rate", fmt(s["H15"]), p3(s["H15"]["p"]), "--"]]
    tex(out / "v4_hypotheses.tex", ["hypothesis", "contrast", r"estimate [95\,\% CI]", "$p$", "Holm $p$"], rows,
        f"SAP_v4.md H11-H15: differences of the seed ensembles, {s['n_boot']} bootstrap resamples (NIH: patient clusters; H14: scans; H15: "
        "patients); Holm over H11 and the two H12 contrasts")
    sh = s["shared_auroc"]
    tex(out / "v4_macro.tex", ["model", "seeds", "NIH macro-AUROC (14 classes)", "mean AUROC, 5 shared findings"],
        [[MODELS[m], str(s["nih_seeds"][m]), f"{v['macro']:.4f}", f"{sum(sh[m]) / len(sh[m]):.4f}" if m in sh else "--"]
         for m, v in s["nih_macro_auroc"].items()],
        "NIH ChestX-ray14 test split, seed ensembles (mean of the seeds' probabilities), descriptive")
    d = json.loads((v4 / "delong.json").read_text())
    pc = s["nih_macro_auroc"]
    pos = np.nansum(np.load(next((v4 / "down" / "nih").glob("nih_hyb_orig@*_s*/test_labels.npy"))), 0)
    tex(out / "v4_per_class.tex", ["class", "test positives", *[MODELS[m] for m in pc], r"$\Delta$ F$_\mathrm{in}$ $-$ DenseNet-121", "Holm $p$"],
        [[c.replace("_", " "), f"{int(pos[i]):,}"] + [f"{pc[m]['per_class'][i]:.4f}" for m in pc] + [f"{d['nih14'][f'F_in-DN121:{c}']['diff']:+.4f}",
                                                                               p3(d["nih14"][f"F_in-DN121:{c}"]["p_holm"])]
         for i, c in enumerate(NIH)],
        "NIH per-class test AUROC of the seed ensembles; paired DeLong test F_in against DenseNet-121, Holm over the 14 classes (descriptive)")
    con = [("F_p1", "F_in"), ("F_p1", "F_p0"), ("F_p0", "F_in")]
    tex(out / "v4_shared_delong.tex", ["finding", *[MODELS[m] for m in ("F_in", "F_p0", "F_p1")],
                                       *[f"$\\Delta$ {MODELS[a]} $-$ {MODELS[b]} (Holm $p$)" for a, b in con]],
        [[FINDING[f]] + [f"{sh[m][i]:.4f}" for m in ("F_in", "F_p0", "F_p1")]
         + [f"{d['shared'][f'{a}-{b}:{f}']['diff']:+.4f} ({p3(d['shared'][f'{a}-{b}:{f}']['p_holm'])})" for a, b in con]
         for i, f in enumerate(EXACT_NIH)],
        "NIH test AUROC of the shared findings (seed ensembles); paired DeLong tests, Holm over the 15 tests (descriptive)")
    n = {k: len(np.load(next((v4 / "down" / k).glob(f"{k}_xm_*_s*/test_idx.npy")))) for k in ("ct", "qp", "nih", "vin")}
    ev = [("CT teacher", f"CT subset test volumes ({n['ct']:,})", s["ct_teacher"]["ct"])]
    ev += [(f"{m}", f"projection test split ({n['qp']:,})", s["projection"][v]) for m, v in (("P0", "xray"), ("P1", "xraykd"))]
    ev += [(f"{m}, zero-shot", f"{dl} test split ({n[ds]:,})", s["transfer"][ds][v]) for ds, dl in (("nih", "NIH"), ("vin", "VinBigData"))
           for m, v in (("P0", "xray"), ("P1", "xraykd"))]
    tex(out / "v4_transfer.tex", ["model", "evaluation set", "seeds", "macro-AUROC", *[FINDING[f] for f in SHARED]],
        [[n, e, str(r["seeds"]), f"{r['macro']:.4f}", *[f"{a:.3f}" for a in r["per_class"]]] for n, e, r in ev],
        "Seed-ensemble AUROC on the six shared findings (descriptive); NIH lung opacity is the approximate mapping to Infiltration")
    rows = [[FINDING[f], str(cm["per_finding"][f]["n"]), f"{cm['per_finding'][f]['matched']:+.3f}", f"{cm['per_finding'][f]['mismatched']:+.3f}",
             f"{cm0['per_finding'][f]['matched']:+.3f}", f"{cm0['per_finding'][f]['mismatched']:+.3f}"] for f in cm["per_finding"]]
    rows.append(["all (H14)", str(cm["n_maps"]), f"{cm['matched']:+.3f}", f"{cm['mismatched']:+.3f}", f"{cm0['matched']:+.3f}",
                 f"{cm0['mismatched']:+.3f}"])
    tex(out / "v4_crossmodal.tex", ["finding", "maps", "P1 matched", "P1 mismatched", "P0 matched", "P0 mismatched"], rows,
        f"Mean Pearson correlation of the projected CT map with the X-ray map of the same scan (matched) and of the other scans positive for "
        f"the same finding (mismatched); {cm['n_scans']} CT subset test scans, maps averaged over {cm['seeds']} seed pairs; not mirrored "
        f"(frame correlation {cm['frame_r']:.3f}, mirrored {cm['frame_r_flip']:.3f})")
    h = s["hallucination"]
    tex(out / "v4_halluc.tex", ["model", "seeds", "clean", "rib emphasis", "noise", "gamma", "nodule sensitivity", "off-target Grad-CAM++"],
        [[MODELS[m], str(v["seeds"]), *[f"{v[k]:.3f}" for k in ("rate_clean", "rate_rib", "rate_noise", "rate_gamma", "nodule_sensitivity",
                                                               "off_target")]] for m, v in h.items()],
        f"Hallucination tests on the {next(iter(h.values()))['n_free']:,} NIH test images without a finding: share of images with some class "
        "above its threshold (clean) or crossing it after the perturbation; inserted-nodule sensitivity; share of Grad-CAM++ mass outside the "
        "lung field for positive predictions of lung findings")
    tex(out / "v4_nodule_size.tex", ["model", "inserted nodules", *[f"$\\Delta$ score, {k}" for k in next(iter(h.values()))["nodule_dp_by_size"]]],
        [[MODELS[m], f"{v['n_nodule']:,}", *[f"{x:+.5f}" for x in v["nodule_dp_by_size"].values()]] for m, v in h.items()],
        "Counterfactual nodules: mean change of the Mass/Nodule score after insertion by nodule-size tertile")
    pt = json.loads((v4 / "pointing_summary.json").read_text())
    tex(out / "v4_pointing.tex", ["model", "map", r"all targets [95\,\% CI]", "focal", "other"],
        [[MODELS[m], MAPS[k], f"{r['all']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]", f"{r['focal']:.3f}", f"{r['other']:.3f}"]
         for m, v in pt.items() if isinstance(v, dict) for k, r in v["maps"].items() if k in MAPS],
        f"Pointing game on the {next(iter(pt.values()))['n_targets']} NIH expert boxes ({next(iter(pt.values()))['n_focal']} focal: Mass, "
        "Nodule): hit rate averaged over the five seeds, image-cluster bootstrap interval (descriptive)")
    uq = {m: json.loads((v4 / f"uq_cls_{m}.json").read_text())["nih_orig"] for m in ("F_in", "F_p0", "F_p1")}
    tex(out / "v4_conformal.tex", ["model", r"$\lambda$", "test FNR (nominal 0.10)", "mean set size", "AURC (Brier)"],
        [[MODELS[m], f"{u['lam']:.3f}", f"{u['fnr_test']:.3f}", f"{u['mean_set_size']:.2f}", f"{u['aurc_brier']:.4f}"] for m, u in uq.items()],
        f"Conformal risk control of the per-image false-negative rate, calibrated on {uq['F_in']['n_cal']:,} NIH validation images, tested on "
        f"{uq['F_in']['n_test']:,} test images; selective classification by 1 - entropy")
    rows = []
    for name, d, pat, key in (("P0 (projections)", "qp", "qp_xm_xray_s*", "test_macro_auc"), ("P1 (projections)", "qp", "qp_xm_xraykd_s*", "test_macro_auc"),
                              ("P0 zero-shot, NIH", "nih", "nih_xm_transfer-xray_s*", "mean"), ("P1 zero-shot, NIH", "nih", "nih_xm_transfer-xraykd_s*", "mean"),
                              ("P0 zero-shot, VinBigData", "vin", "vin_xm_transfer-xray_s*", "mean"),
                              ("P1 zero-shot, VinBigData", "vin", "vin_xm_transfer-xraykd_s*", "mean"),
                              ("F_in (NIH)", "nih", "nih_hyb_orig@swinv2_tiny_window8_256_s*", "test_macro_auc"),
                              ("F_p0 (NIH)", "nih", "nih_hyb_orig@swinv2_tiny_window8_256+xray_s*", "test_macro_auc"),
                              ("F_p1 (NIH)", "nih", "nih_hyb_orig@swinv2_tiny_window8_256+xraykd_s*", "test_macro_auc")):
        v = seeds(v4 / "down" / d, pat, key)
        rows.append([name.replace("F_in", MODELS["F_in"]).replace("F_p0", MODELS["F_p0"]).replace("F_p1", MODELS["F_p1"]),
                     *[f"{v[k]:.4f}" if k in v else "--" for k in range(5)]])
    tex(out / "v4_seeds.tex", ["model (test set)", *[f"seed {k}" for k in range(5)]], rows,
        "Test macro-AUROC of each seed's model (single runs; the tables above use the seed ensembles)")
    ct = sorted(Path(ct_runs).glob("ct-*/results/ct_xm_ct_s*/result.json")) if ct_runs else []
    if ct:
        r = {int(f.parent.name.rsplit("_s", 1)[1]): json.loads(f.read_text()) for f in ct}
        tex(out / "v4_ct_seeds.tex", ["CT teacher seed", "best validation macro-AUROC", "test macro-AUROC", "use"],
            [[str(k), f"{v['val_macro_auc']:.4f}", f"{v['test_macro_auc']:.4f}", CT_SLOT[k]] for k, v in sorted(r.items())],
            "Every CT teacher run (20 epochs, unchanged recipe); slot k is the teacher of P1 seed k (DEVIATIONS.md, 9 Nov)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=str(ROOT / "results/final_xm_stats.json"))
    ap.add_argument("--v4", default=str(ROOT / "results/v4"))
    ap.add_argument("--ct_runs", default=str(ROOT / "outputs/v4"))
    ap.add_argument("--tables", default=str(ROOT / "paper/tables"))
    a = ap.parse_args(argv)
    write_tables(json.loads(Path(a.stats).read_text()), Path(a.v4), a.ct_runs, Path(a.tables))
    print("wrote", ", ".join(sorted(p.name for p in Path(a.tables).glob("v4_*.tex"))))


if __name__ == "__main__":
    main()
