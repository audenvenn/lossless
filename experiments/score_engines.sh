#!/bin/bash
# Score every engine pair (spec arm vs independent-seed off arm) with one fixed reference; same-seed identity checks.
set -u
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
inner() {
  for S in qwen_think_T0.6_p0.95_k20 T1.0_plain; do
    for pair in "kapitsa mtp off" "kapitsa dflash off" "mtplx mtp off" "raceline on off2"; do
      set -- $pair; E=$1; A1=$2; A0=$3
      f1=results/l3/${E}_${S}_arm${A1}.jsonl; f0=results/l3/${E}_${S}_arm${A0}.jsonl
      [ -s "$f1" ] && [ -s "$f0" ] || { echo "skip $E $A1/$A0 $S"; continue; }
      $PY -I experiments/l3_score.py --arm-files $f1 $f0 --setting $S --out results/l3/score_${E}_${A1}_vs_${A0}_${S}.json \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('$E $A1 vs $A0 $S', {k: round(v,4) for k,v in d['p'].items()}, 'len', round(d['mean_len_arm1'],1), round(d['mean_len_arm0'],1))"
    done
    f1=results/l3/raceline_${S}_armon.jsonl; f0=results/l3/raceline_${S}_armoffsame.jsonl
    [ -s "$f1" ] && [ -s "$f0" ] && $PY -I experiments/same_seed_identity.py $f1 $f0 --out results/l3/identity_raceline_${S}.json | head -8
  done
}
export -f inner; export PY
GPU_LOCK_WAIT_MAX=7200 "${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}" "L3 scoring of engine pairs (MLX 4B reference, prefill only), ~15 min, ~4 GB" bash -c inner
