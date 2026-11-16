#!/bin/bash
# Batch 2 of the v0.4 revision (after run_r2x.sh). Each step holds the shared GPU lock only while it runs.
#  1) cache fixed-reference scores (MLX 4-bit) for every audit arm file and every R2/R2x sequence file
#  2) cache bf16-reference scores for R2/R2x
#  3) near-tie analysis at greedy divergence points (bf16)          4) analytic first-token expectation (4-bit)
#  5) llama.cpp spec-off arm with PRE-sampling top-20 log-probs (L1 against the declared llama.cpp semantics)
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
PY=../../.venv/bin/python
LOCK="${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}"
export GPU_LOCK_WAIT_MAX=10800
$LOCK "batch2: L3 reference scoring (MLX 4-bit Qwen3.5-4B, prefill only), ~10 min, ~4 GB" \
  nice -n 5 $PY -I experiments/score_arms.py --ref "$LOSSLESS_QWEN_MLX4" --out-dir results/zref/mlx4 \
  results/l3/*.jsonl results/r2/*.seqs.json results/r2x/*.seqs.json 2>&1 | grep -vE "Warning"
$LOCK "batch2: R2 reference scoring (bf16 Qwen3.5-4B, prefill only), ~6 min, ~10 GB" \
  nice -n 5 $PY -I experiments/score_arms.py --ref "$LOSSLESS_QWEN_BF16" --out-dir results/zref/bf16 \
  results/r2/*.seqs.json results/r2x/*.seqs.json 2>&1 | grep -vE "Warning"
$LOCK "batch2: near-tie prefill at greedy divergences (bf16), ~2 min, ~10 GB" \
  nice -n 5 $PY -I experiments/r3_neartie.py 2>&1 | grep -vE "Warning"
$LOCK "batch2: first-token distribution (MLX 4-bit), ~1 min, ~4 GB" \
  nice -n 5 $PY -I experiments/first_token_expected.py 2>&1 | grep -vE "Warning"
# llama.cpp: pre-sampling probabilities (no post_sampling_probs), independent seed, spec off only
G=engines/gguf/qwen3.5-4b-q8_0.gguf
mkdir -p engines/gguf
if [ ! -s "$G" ]; then
  $LOCK "batch2: GGUF conversion Qwen3.5-4B q8_0 (CPU/IO heavy), ~10 min, ~6 GB" \
    nice -n 5 engines/.venv-convert/bin/python engines/llama-convert/convert_hf_to_gguf.py \
    "$LOSSLESS_QWEN_BF16" --outtype q8_0 --outfile "$G" 2>&1 | tail -3
fi
if [ -s "$G" ]; then
  EXTRA='' SEED=3000000 experiments/run_l3.sh llamacppraw off 8814 /health qwen -- \
    llama-server -m $G --port 8814 --host 127.0.0.1 -ngl 999 -fa on --jinja -c 8192 --spec-type none
  rm -f "$G" && echo "GGUF removed (disk)"
fi
echo "batch2 done"
