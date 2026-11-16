#!/bin/bash
# After run_blind.sh: fixed-reference scores for the four pre-registered commits (GPU, prefill only), then CPU analyses.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
PY=../../.venv/bin/python
LOCK="${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}"
export GPU_LOCK_WAIT_MAX=10800
$LOCK "R2b scoring (bf16 + MLX 4-bit refs, prefill only), ~5 min, ~10 GB" bash -c "
  nice -n 5 $PY -I experiments/score_arms.py --ref \"$LOSSLESS_QWEN_BF16\" --out-dir results/zref/bf16 results/r2b/*.seqs.json &&
  nice -n 5 $PY -I experiments/score_arms.py --ref \"$LOSSLESS_QWEN_MLX4\" --out-dir results/zref/mlx4 results/r2b/*.seqs.json" 2>&1 | grep -v Warning
nice -n 10 $PY -I experiments/r2x_analyze.py results/r2b 2>&1 | grep -v Warning
nice -n 10 $PY -I experiments/l3_compare.py experiments/pairs_blind.json results/l3_blind.json 2>&1 | grep -v Warning
echo "blind post done"
