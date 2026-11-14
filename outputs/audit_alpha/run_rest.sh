#!/bin/bash
cd "C:/Thesis new/swintrust4"
while [ ! -f outputs/audit_alpha/clia_s0.csv ]; do sleep 15; done
python -m s4.audit_real --lib outputs/lesions_test.npz --methods ours:outputs/scr_all/weights/qp_s0/final_ema.pt --alphas 0.25 0.5 1 2 4 --out outputs/audit_alpha/qp_s0.csv > outputs/audit_alpha/qp.log 2>&1
python -m s4.audit_real --lib outputs/lesions_test.npz --methods qure:data/qure/weights/bone_suppression.ts --alphas 0.25 0.5 1 2 4 --out outputs/audit_alpha/qure.csv > outputs/audit_alpha/qure.log 2>&1
echo ALPHA DONE
