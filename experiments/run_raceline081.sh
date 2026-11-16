#!/bin/bash
# raceline 0.8.1 (sampler tail fix): same-seed identity MTP on vs off, both settings.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
R=experiments/run_l3.sh
RL=${RACELINE_BIN:?set RACELINE_BIN (see paths.local.env)}
F=${RACELINE_MODEL:?set RACELINE_MODEL}
SEED=1000 $R raceline081 on      8813 /v1/models local -- $RL serve $F --port 8813
SEED=1000 $R raceline081 offsame 8813 /v1/models local -- env RACELINE_MTP=0 $RL serve $F --port 8813
echo "raceline081 done"
