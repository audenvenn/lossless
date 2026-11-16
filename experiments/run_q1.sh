#!/bin/bash
# Q1: does the small positive surprisal z of D-head spec-on at T=1 grow with N? 48 prompts x 3 samples.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
ENG=engines/mlx-lm-mtp; PY=engines/.venv-mtp/bin/python; LOCK="${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}"
(cd $ENG && git checkout -q -f c4963b1b9e)
for spec in on off; do
  GPU_LOCK_WAIT_MAX=7200 $LOCK "Q1 D-head T=1 spec-$spec x3 samples, bf16 4B, ~30 min, ~10 GB" \
    nice -n 5 $PY -I experiments/r2_mlx_mtp.py --label Q1_D_head --spec $spec --setting T1.0_plain --n-prompts 48 --samples 3 \
    --max-tokens 256 --seed 99 --out results/q1 2>&1 | grep -vE "Warning|z = -_Ak"
done
echo "Q1 done"
