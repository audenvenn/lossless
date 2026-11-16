#!/bin/bash
# R3: greedy-identity column of the natural experiment (temp 0), A-D spec-on vs D spec-off.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
ENG=engines/mlx-lm-mtp; PY=engines/.venv-mtp/bin/python; LOCK="${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}"
run() { local label=$1 commit=$2 spec=$3
  (cd $ENG && git checkout -q -f $commit) || return 1
  GPU_LOCK_WAIT_MAX=7200 $LOCK "R3 greedy $label spec-$spec, bf16 4B, ~8 min, ~10 GB" \
    nice -n 5 $PY -I experiments/r2_mlx_mtp.py --label $label --spec $spec --setting greedy --n-prompts 48 --max-tokens 256 --out results/r3 2>&1 | grep -vE "Warning|z = -_Ak"
  (cd $ENG && git checkout -q -f c4963b1b9e)
}
run D_head c4963b1b9e off
run D_head c4963b1b9e on
run A_initial 1ff33d56a0 on
run B_preResid 673b3044da on
run C_preLpAcc 96093cfae0 on
echo "R3 done"
