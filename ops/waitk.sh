#!/bin/bash
# helpers: wait_ds <ac> <user> <slug> (until the dataset is ready); wait_k <ac> <user> <kernel> (until it leaves RUNNING/QUEUED; prints the end state)
cd "$(dirname "$0")/.."
wait_ds() { until python ops/kg.py $1 datasets status $2/$3 2>&1 | grep -q ready; do sleep 60; done; echo "$3 ready on $1"; }
wait_k() {
  while true; do
    s=$(python ops/kg.py $1 kernels status $2/$3 2>&1 | grep -a -o 'KernelWorkerStatus\.[A-Z]*')
    case "$s" in *RUNNING*|*QUEUED*|"") sleep 120 ;; *) echo "$3 $s"; return ;; esac
  done
}
