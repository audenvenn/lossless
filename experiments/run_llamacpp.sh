#!/bin/bash
# llama.cpp (brew 0.9.0, b12110): Qwen3.5-4B Q8_0 GGUF with the MTP head; draft-mtp vs no speculation.
# 1) convert under the lock (heavy read of the 8.7 GB checkpoint), 2) three arms, logprobs with post-sampling probs.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
G=engines/gguf/qwen3.5-4b-q8_0.gguf
mkdir -p engines/gguf
if [ ! -s "$G" ]; then
  df -h /System/Volumes/Data | tail -1
  GPU_LOCK_WAIT_MAX=7200 "${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}" "GGUF conversion Qwen3.5-4B q8_0 (CPU/IO heavy), ~10 min, ~6 GB" \
    nice -n 5 engines/.venv-convert/bin/python engines/llama-convert/convert_hf_to_gguf.py \
    "${LOSSLESS_QWEN_BF16:?set LOSSLESS_QWEN_BF16 (HF checkpoint dir with the MTP layer)}" --outtype q8_0 --outfile "$G" 2>&1 | tail -5
fi
[ -s "$G" ] || { echo "conversion failed"; exit 1; }
R=experiments/run_l3.sh
L="llama-server -m $G --port 8814 --host 127.0.0.1 -ngl 999 -fa on --jinja -c 8192"
export EXTRA='{"post_sampling_probs": true}'
SEED=2000000 $R llamacpp off     8814 /health qwen -- $L --spec-type none
SEED=1000    $R llamacpp offsame 8814 /health qwen -- $L --spec-type none
SEED=1000    $R llamacpp mtp     8814 /health qwen -- $L --spec-type draft-mtp
echo "llamacpp done"
