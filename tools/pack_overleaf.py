"""Package the combined manuscript (submission/) as one Overleaf project: copy every table and figure that the sections reference from the
generated outputs (paper/tables, paper/figures, results/figures), then zip main.tex, sections/, tables/, figures/, refs.bib and the text files.
python tools/pack_overleaf.py            -> overleaf_submission.zip in the repository root (refuses if a referenced file is missing)"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
SOURCES = {"tables": [ROOT / "paper/tables"], "figures": [ROOT / "paper/figures", ROOT / "results/figures"]}

TEX = [SUB / "main.tex", SUB / "supplementary.tex", *sorted((SUB / "sections").glob("*.tex"))]
tex = "\n".join(p.read_text(encoding="utf-8") for p in TEX)
tex = "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in tex.splitlines())
wanted = sorted(set(re.findall(r"\\(?:fittable|input)\{(tables/[^}]+)\}", tex)) | set(re.findall(r"\{(figures/[^}]+)\}", tex)))
missing = []
for rel in wanted:
    kind, name = rel.split("/", 1)
    src = next((d / name for d in SOURCES[kind] if (d / name).is_file()), None)
    if src is None:
        missing.append(rel)
        continue
    (SUB / kind).mkdir(exist_ok=True)
    (SUB / rel).write_bytes(src.read_bytes())
assert not missing, f"referenced but not found: {missing}"
files = [*TEX, SUB / "refs.bib", SUB / "highlights.txt", SUB / "README.txt", SUB / "REVIEW_Q1.md", *(SUB / rel for rel in wanted)]
out = ROOT / "overleaf_submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(SUB).as_posix())
print(f"{out.name}: {len(files)} files, {out.stat().st_size / 1e6:.1f} MB ({len(wanted)} tables and figures copied)")
