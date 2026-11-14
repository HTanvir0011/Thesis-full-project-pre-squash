"""Build s4/data/xray_tables.json: ICRU-44 mass attenuation (NIST SRD 126, Hubbell & Seltzer, table 4) + SpekPy tube spectra.
Raw NIST pages are kept in s4/data/nist_raw/ for provenance. Run once; the JSON is committed."""
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import spekpy

OUT = Path(__file__).resolve().parents[1] / "s4" / "data"
RAW = OUT / "nist_raw"
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/"
MATERIALS = {"bone_cortical": "bone.html", "soft_tissue": "tissue.html", "lung": "lung.html"}
KVPS = [60, 70, 80, 90, 100, 110, 120, 130, 140]
E = np.arange(10.0, 151.0, 1.0)  # keV grid, 1 keV


def nist_rows(name):
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0 (swintrust4 research script)"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("latin-1")
    (RAW / name).write_text(html, encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", html).replace("&nbsp;", " ")
    nums = re.findall(r"[-+]?\d\.\d+E[-+]\d+", text)
    assert len(nums) % 3 == 0 and len(nums) > 90, f"{name}: unexpected number count {len(nums)}"
    a = np.array(nums, float).reshape(-1, 3)  # MeV, mu/rho, muen/rho
    return a[:, 0] * 1e3, a[:, 1]


tables = {"energy_keV": E.tolist(), "mu_over_rho_cm2_per_g": {}, "spectra": {}}
for mat, page in MATERIALS.items():
    ek, mu = nist_rows(page)
    keep = (ek >= 8) & (ek <= 200)
    ek, mu = ek[keep], mu[keep]
    ek = ek + np.arange(len(ek)) * 1e-6  # absorption-edge rows share an energy; keep them monotone
    tables["mu_over_rho_cm2_per_g"][mat] = np.exp(np.interp(np.log(E), np.log(ek), np.log(mu))).tolist()

for kvp in KVPS:
    s = spekpy.Spek(kvp=kvp, th=12, dk=1, physics="kqp")
    s.filter("Al", 2.5)
    k, phi = s.get_spectrum(edges=False)
    phi = np.interp(E, k, phi, left=0.0, right=0.0)
    tables["spectra"][str(kvp)] = (phi / phi.sum()).tolist()

tables["provenance"] = {
    "attenuation": "NIST SRD 126 X-ray mass attenuation coefficients (Hubbell & Seltzer), Table 4 (ICRU-44 materials), pages in nist_raw/, log-log interpolated to 1 keV grid",
    "spectra": f"SpekPy {getattr(spekpy, '__version__', '?')}, W anode 12 deg, kqp, 2.5 mm Al",
    "detector": "energy-integrating: weight = phi(E)*E",
}
(OUT / "xray_tables.json").write_text(json.dumps(tables))
print("wrote", OUT / "xray_tables.json", {m: round(tables["mu_over_rho_cm2_per_g"][m][50 - 10], 4) for m in MATERIALS})
