#!/bin/bash
# Parallel driver for tools/final_local.sh: one worker per method (all of its steps), N workers share the one GPU (the evaluations are mostly CPU-bound).
# usage: [N=3] bash tools/final_parallel.sh    Resumable (existing outputs are kept). The H5' uncertainty step is not per method, so it runs once at the end.
# Rule R4: final_local.sh refuses to run before protocol/freeze_v2.json exists.
cd "$(dirname "$0")/.."
[ -f protocol/freeze_v2.json ] || { echo "protocol/freeze_v2.json missing: run tools/freeze_v2.py first (RUNPLAN.md R4)"; exit 1; }
grep -v '^#' tools/final_methods.txt | grep '=' | cut -d= -f1 \
  | xargs -P "${N:-3}" -I{} bash -c 'for s in szch audit rib qp synth jsrt; do bash tools/final_local.sh $s {}; done'
bash tools/final_local.sh unc
