#!/bin/bash
# Evaluate every weights dir without val results on the synthetic VAL CTs (model selection uses val only).
cd "C:/Thesis new/swintrust4"
todo=()
for d in outputs/abl_all/weights/*/; do n=$(basename $d); [ -f outputs/abl_all/val/$n/per_site.csv ] || todo+=("$d/final_ema.pt"); done
[ ${#todo[@]} -eq 0 ] && { echo nothing to do; exit 0; }
python -m s4.eval_synth --ckpt "${todo[@]}" --bank outputs/shard0/render outputs/shard1/render outputs/shard2/render --split splits/ct_split.csv --which val --out outputs/abl_all/val
