#!/bin/bash
# Run one engine configuration as an OpenAI-compatible server (under the GPU lock), collect L3 replies for
# the given settings, stop the server. Usage:
#   experiments/run_l3.sh <label> <arm> <port> <health_path> <api_model> -- <server command...>
# Env: SETS (space-separated settings), N_PROMPTS, SAMPLES, MAX_TOKENS
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
LABEL=$1; ARM=$2; PORT=$3; HEALTH=$4; MODEL=$5; shift 5
[ "$1" = "--" ] && shift
SETS=${SETS:-"qwen_think_T0.6_p0.95_k20 T1.0_plain"}
N=${N_PROMPTS:-32}; S=${SAMPLES:-2}; L=${MAX_TOKENS:-256}; SEED=${SEED:-1000}; EXTRA=${EXTRA:-}
PY=../../.venv/bin/python
inner() {
  "$@" > results/l3/${LABEL}_arm${ARM}.server.log 2>&1 &
  SRV=$!
  for i in $(seq 1 600); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT$HEALTH" > /dev/null 2>&1; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "[run_l3] server died"; tail -20 results/l3/${LABEL}_arm${ARM}.server.log; return 1; fi
    sleep 1
  done
  echo "[run_l3] $LABEL arm=$ARM up after ${i}s"
  for SET in $SETS; do
    $PY -I experiments/l3_client.py --base-url http://127.0.0.1:$PORT --model "$MODEL" --label $LABEL --arm $ARM \
      --setting $SET --n-prompts $N --samples $S --max-tokens $L --seed $SEED ${EXTRA:+--extra-body "$EXTRA"} --out results/l3 || echo "[run_l3] client failed for $SET"
  done
  kill -TERM $SRV 2>/dev/null; sleep 3; kill -KILL $SRV 2>/dev/null; wait $SRV 2>/dev/null
  return 0
}
mkdir -p results/l3
export -f inner
export LABEL ARM PORT HEALTH MODEL SETS N S L PY SEED EXTRA
GPU_LOCK_WAIT_MAX=7200 "${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}" "L3 $LABEL arm=$ARM (4B engine server), ~25 min, ~6 GB" bash -c 'inner "$@"' _ "$@"
