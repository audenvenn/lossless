#!/bin/bash
# raceline (the user's engine; disclose in any write-up): MTP on vs off.
#  on   seed 1000      | off_same seed 1000  -> same-seed identity ("bit-for-bit at any temperature")
#  off2 seed 2000000   -> independent arm for the exact two-sample test (on vs off2)
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
R=experiments/run_l3.sh
RL=${RACELINE_BIN:?set RACELINE_BIN (see paths.local.env)}
F=${RACELINE_MODEL:?set RACELINE_MODEL}
SEED=1000    $R raceline on       8813 /v1/models local -- $RL serve $F --port 8813
SEED=1000    $R raceline offsame  8813 /v1/models local -- env RACELINE_MTP=0 $RL serve $F --port 8813
SEED=2000000 $R raceline off2     8813 /v1/models local -- env RACELINE_MTP=0 $RL serve $F --port 8813
echo "raceline done"
