#!/bin/bash
# L3 arms for kapitsa and MTPLX on Qwen3.5-4B. Independent seeds per arm (two-sample test); a separate small
# same-seed run checks the "changes no output" claims. Each arm holds the GPU lock only while its server runs.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
R=experiments/run_l3.sh
K="kapitsa serve --model 4b --port 8811 --private --local"
M="${MTPLX_BIN:?set MTPLX_BIN} serve --model Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed --port 8812 --no-auth --yes"
MM="Youssofal/Qwen3.5-4B-MTPLX-Optimized-Speed"
# kapitsa: off (no speculation), MTP head, default (DFlash drafter)
SEED=2000000 $R kapitsa off   8811 /v1/models qwen3.5-4b -- $K --spec 0 --dflash none
SEED=1000    $R kapitsa mtp   8811 /v1/models qwen3.5-4b -- $K --dflash none
SEED=1000    $R kapitsa dflash 8811 /v1/models qwen3.5-4b -- $K
# MTPLX: AR (no MTP) vs MTP (default profile)
SEED=2000000 $R mtplx off 8812 /health $MM -- $M --no-mtp
SEED=1000    $R mtplx mtp 8812 /health $MM -- $M --mtp
echo "engines done"
