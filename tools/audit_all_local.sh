#!/bin/bash
# Real-image lesion audit (SZCH, 450 sites) for every weights dir under outputs/abl_all/weights that has no audit csv yet.
cd "C:/Thesis new/swintrust4"
mkdir -p outputs/audit_final
for d in outputs/abl_all/weights/*/; do
  n=$(basename $d)
  [ -f outputs/audit_final/$n.csv ] && continue
  python -m s4.audit_real --lib outputs/lesions_test.npz --methods ours:$d/final_ema.pt --out outputs/audit_final/$n.csv > outputs/audit_final_$n.log 2>&1
done
