#!/bin/bash
# Pre-registered validation on four more PR #990 commits (results/blind_commits_plan.json, written before any run):
# thinking setting, white-box exact L1 (capture v2.1), independent seeds 301-304; L3 later against D-off seed 105.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
ENG=engines/mlx-lm-mtp
PY=engines/.venv-mtp/bin/python
LOCK="${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}"
N=${N_PROMPTS:-48}; L=${MAX_TOKENS:-256}; OUT=${OUT:-results/r2b}
TH=qwen_think_T0.6_p0.95_k20
run() { # label commit patch-sed-expr seed
  local label=$1 commit=$2 expr=$3 seed=$4
  (cd $ENG && git checkout -q -f $commit) || return 1
  if [ -n "$expr" ]; then
    sed -i '' "$expr" $ENG/mlx_lm/generate.py
    n=$(grep -c "verify_lp, True" $ENG/mlx_lm/generate.py); echo "[$label] reporting patch applied: $n site(s)"
  fi
  GPU_LOCK_WAIT_MAX=${GPU_LOCK_WAIT_MAX:-10800} $LOCK "R2b $label $TH spec-on seed $seed, bf16 Qwen3.5-4B, ~11 min, ~10 GB" \
    nice -n 5 $PY -I experiments/r2_mlx_mtp.py --label $label --spec on --setting $TH --seed $seed \
    --n-prompts $N --max-tokens $L --out $OUT 2>&1 | grep -vE "Warning|z = -_Ak"
  (cd $ENG && git checkout -q -f c4963b1b9e)
}
if [ "${SMOKE:-0}" = 1 ]; then run SMOKE_D c4963b1b9e "" 399; exit 0; fi
run X1_2555209 2555209 's/yield draft_tok.item(), draft_lp, True/yield draft_tok.item(), verify_lp, True/' 301
run X2_d737eed d737eed 's/yield draft_tok_id, draft_lp, True/yield draft_tok_id, verify_lp, True/'       302
run X3_e8a8504 e8a8504 's/yield draft_tok_id, draft_lp, True/yield draft_tok_id, verify_lp, True/'       303
run X4_dcbb45a dcbb45a ''                                                                                304
echo "R2b done"
