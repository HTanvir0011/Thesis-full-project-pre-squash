"""Build kdata/s4-code/ (the s4 package + series lists + LIDC XML) so a Kaggle kernel can import it as a dataset."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "kdata" / "s4-code" / "code"  # Kaggle flattens the top-level dir of a dataset, so s4/ lives one level down
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(ROOT / "s4", DST / "s4", ignore=shutil.ignore_patterns("__pycache__", "nist_raw"))
data = DST / "s4" / "data"
shutil.copy(ROOT / "outputs" / "tcia_lidc_ct_series.csv", data / "tcia_lidc_ct_series.csv")
shutil.copy(ROOT / "outputs" / "lidc_audit" / "lidc_series.csv", data / "lidc_mount_series.csv")
shutil.copytree(ROOT / "data" / "lidc_xml", data / "lidc_xml")
shutil.copy(ROOT / "outputs" / "lib_train.npz", data / "lib_train.npz")  # lesion library of the train CTs (tools/make_lib_img.py): bank-free QP arms
shutil.copy(ROOT / "results" / "qp_meta.csv", data / "qp_meta.csv")  # split + bone weight per Qure projection row; the CPU pack kernel (kernels/prep_qp) reads it
BS = ROOT.parent / "third_party" / "bsldm_src"  # BS-LDM sources (MIT, github.com/diaoquesang/BS-LDM): the baseline kernel copies them next to the downloaded weights
if BS.exists():
    shutil.copytree(BS, data / "bsldm_src", ignore=shutil.ignore_patterns("__pycache__"))
EV = DST / "evalpack"  # locked-set evaluation on Kaggle (kernels/evalk): tools/final_local.sh runs unchanged inside a workspace built from this
(EV / "tools").mkdir(parents=True)
for f in ("final_local.sh", "final_methods.txt"):
    shutil.copy(ROOT / "tools" / f, EV / "tools" / f)
(EV / "protocol").mkdir()
shutil.copy(ROOT / "protocol" / "freeze_v2.json", EV / "protocol" / "freeze_v2.json")
(EV / "splits").mkdir()
shutil.copy(ROOT / "splits" / "ct_split.csv", EV / "splits" / "ct_split.csv")
shutil.copytree(ROOT / "outputs" / "audit_masks", EV / "audit_masks", ignore=shutil.ignore_patterns("*ribcxr*", "*RibCXR*"))  # local lung/bone masks: Kaggle audits draw the local sites (tools/export_audit_masks.py); licensed RibCXR masks never ship
(DST / "tools").mkdir()
shutil.copy(ROOT / "tools" / "szch_split.py", DST / "tools" / "szch_split.py")  # s4/eval_uncertainty.py imports it from <code>/tools
assert not [p for p in DST.rglob("*") if "ribcxr" in p.name.lower() and p.suffix != ".py"], "licensed RibCXR data must not ship in s4-code (ac_1 only)"
print("packaged", sum(1 for _ in DST.rglob("*") if _.is_file()), "files,",
      round(sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1e6, 1), "MB")
