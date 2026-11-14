#!/bin/bash
# Wave 1 (after the 2026-10-31 06:00 refill): 10 training kernels, 2 per account. Accounts run in parallel, launches within an
# account one after another (they share the account's code staging folder). Logs: outputs/launch_<kernel>.log
cd "$(dirname "$0")/.."
python tools/package_code.py
launch() {  # account, then "kernel flags" pairs
  local ac=$1; shift
  for k in "$@"; do
    set -- $k
    python ops/launch_train.py $ac $1 NvidiaTeslaT4 outputs/jobs/$1.json ${@:2} > outputs/launch_$1.log 2>&1
    grep -a "successfully pushed\|rror" outputs/launch_$1.log | tail -1
  done
}
launch ac_1 "s4-w1-1" "s4-w1-2 --qp" &
launch ac_2 "s4-w1-3 --qp" "s4-w1-4 --qp" &
launch ac_3 "s4-w1b-B --qp" "s4-w1-5 --qp" &   # s4-w1-6 follows when a slot frees; s4-w1b-B continues as s4-w1b-B2 (--resume s4-w1b-B)
launch ac_4 "s4-w1-7 --qpk --nobank" "s4-w1-8 --qpk --nobank" &
launch ac_5 "s4-w1-9 --qpk --nobank" "s4-w1-10 --qpk --nobank" &
wait
echo WAVE1 LAUNCHED
