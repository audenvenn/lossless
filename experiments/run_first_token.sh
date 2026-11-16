#!/bin/bash
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
PY=../../.venv/bin/python
inner() {
  local name=$1 port=$2 health=$3 model=$4; shift 4
  "$@" > results/first_token_${name}.server.log 2>&1 &
  local SRV=$!
  for i in $(seq 1 300); do curl -s -m 2 "http://127.0.0.1:$port$health" >/dev/null 2>&1 && break; sleep 1; done
  $PY -I experiments/first_token_support.py http://127.0.0.1:$port "$model" 400 results/first_token_${name}.json
  kill -TERM $SRV 2>/dev/null; sleep 3; kill -KILL $SRV 2>/dev/null; wait $SRV 2>/dev/null
}
export -f inner; export PY
GPU_LOCK_WAIT_MAX=3600 "${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}" "first-token support repro: kapitsa 4b then raceline 0.8.1 4B, ~10 min, ~6 GB" bash -c '
inner kapitsa 8811 /v1/models qwen3.5-4b kapitsa serve --model 4b --port 8811 --private --local --spec 0 --dflash none
inner raceline081 8813 /v1/models local $RACELINE_BIN serve $RACELINE_MODEL --port 8813
'
