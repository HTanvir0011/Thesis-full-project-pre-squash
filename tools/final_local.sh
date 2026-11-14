#!/bin/bash
# Locked-set evaluation on the local GPU (RUNPLAN.md section 6). Refuses to run before protocol/freeze_v2.json exists (rule R4).
# usage: [DRY=1] bash tools/final_local.sh <szch|audit|rib|jsrt|qp|synth|unc|all> [method-name ...]
# Methods come from tools/final_methods.txt; existing result files are kept (delete one to recompute it). VinDr-RibCXR: metrics only.
set -u
cd "$(dirname "$0")/.."
[ -f protocol/freeze_v2.json ] || [ -n "${DRY:-}" ] || { echo "protocol/freeze_v2.json missing: run tools/freeze_v2.py first (RUNPLAN.md R4)"; exit 1; }
step=$1; shift
want=" $* "
R=results/final; LIB=outputs/lesions_test.npz; QURE=data/qure/weights/bone_suppression.ts
mkdir -p $R
run() {  # step-tag, output path template, command template ({name}, {spec} are filled in)
  local tag=$1 otpl=$2 tpl=$3
  grep -v '^#' tools/final_methods.txt | grep '=' | while IFS= read -r line; do
    name=${line%%=*}; rest=${line#*=}; spec=$(echo "${rest%%|*}" | xargs | sed "s#W/#outputs/final_v2/weights/#g"); tags=" ${rest#*|} "
    [[ "$tags" == *" $tag "* ]] || continue
    [ "$want" = "  " ] || [[ "$want" == *" $name "* ]] || continue
    for f in $(echo "$spec" | sed 's/^[a-z]*://; s/+/ /g'); do [[ "$f" == *.pt || "$f" == *.ts ]] && [ ! -f "$f" ] && { echo "skip $name: missing $f"; continue 2; }; done
    cmd=${tpl//\{name\}/$name}; cmd=${cmd//\{spec\}/$spec}; out=${otpl//\{name\}/$name}
    case $out in *audit_ribcxr*|*/rib/*) [ -d data/ribcxr_raw ] || continue ;; esac  # licensed RibCXR runs only where its data is mounted (ac_1 / local)
    [ -e "$out" ] && { echo "have $out"; continue; }
    [ -n "${DRY:-}" ] && { echo "DRY $out"; continue; }  # DRY=1: list what would run (allowed before the freeze)
    echo "== $name: $cmd"
    eval "$cmd" 2>&1 | grep -v -i "warn\|symlink" | tail -3
    [ -e "$out" ] || echo "Error: $out not written"  # the pipe hides the exit code; kernels/evalk flags this line
  done
}
do_step() {
  case $1 in
    szch)  run szch $R/szch/{name}.csv "python -m s4.eval_real --method {spec} --lpips --out $R/szch/{name}.csv" ;;
    audit) run audit $R/audit_ribcxr/{name}.csv "python -m s4.audit_real --dataset ribcxr --root data/ribcxr_raw --lib $LIB --methods {spec} --out $R/audit_ribcxr/{name}.csv"
           run audit $R/audit_jsrt/{name}.csv "python -m s4.audit_real --dataset jsrt --root data/jsrt/images --lib $LIB --overlap_ref qure:$QURE --methods {spec} --out $R/audit_jsrt/{name}.csv"
           run audit $R/audit_szch/{name}.csv "python -m s4.audit_real --dataset szch --root data/szch --lib $LIB --methods {spec} --out $R/audit_szch/{name}.csv" ;;
    rib)   run rib $R/rib/{name}.csv "python -m s4.eval_ribcxr --root data/ribcxr_raw --method {spec} --out $R/rib/{name}.csv" ;;
    qp)    run qp $R/qp/{name} "python -m s4.audit_qp --pack outputs/qp_pack --lib $LIB --methods {spec} --out $R/qp/{name}" ;;
    synth) run synth $R/synth/{name} "python -m s4.eval_synth --ckpt \$(echo {spec} | sed 's/^[a-z]*://; s/+/ /g') --bank outputs/shard0/render outputs/shard1/render outputs/shard2/render --split splits/ct_split.csv --out $R/synth/{name}" ;;
    jsrt)  run jsrt $R/jsrt/{name} "S4_PREP_OUT=$R/jsrt S4_JSRT=data/jsrt python -m s4.prep --dataset jsrt --weights \$(echo {spec} | sed 's/^[a-z]*://') --qure $QURE --arrays orig,soft,bone,soft_qure --tag {name}" ;;
    unc)   [ -n "${DRY:-}" ] && { [ -e $R/unc_heldout.json ] && echo "have $R/unc_heldout.json" || echo "DRY $R/unc_heldout.json"; return; }
           [ -e $R/unc_heldout.json ] || python -m s4.eval_uncertainty --ckpts outputs/final_v2/weights/qp_twin16_s{0,1,2}_cnx2_300k/final_ema.pt --qp_pack outputs/qp_pack --szch_subset held --out $R/unc_heldout.json 2>&1 | tail -14 ;;
  esac
}
if [ "$step" = all ]; then for s in szch audit rib qp synth jsrt unc; do do_step $s; done; else do_step $step; fi
