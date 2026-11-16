#!/bin/bash
# Same-seed speculation-off arms (seed 1000, as the spec-on arms) for identity checks against plain decoding.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
R=experiments/run_l3.sh
K="kapitsa serve --model 4b --port 8811 --private --local"
M="${MTPLX_BIN:?set MTPLX_BIN} serve --model Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed --port 8812 --no-auth --yes"
MM="Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"
SEED=1000 $R kapitsa offsame 8811 /v1/models qwen3.5-4b -- $K --spec 0 --dflash none
SEED=1000 $R mtplx   offsame 8812 /health   $MM        -- $M --no-mtp
echo "offsame done"
