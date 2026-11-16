#!/bin/bash
# R2x (2026-11-16): regenerate the PR #990 natural experiment with (a) an independent seed per arm and (b) the
# exact white-box L1 capture (engine-own non-speculative transform; see r2_mlx_mtp.py docstring).
# Thinking setting: all five arms. T = 1: a fresh-seed spec-off arm for L3 (the T = 1 L1 results are already exact,
# since no truncation is active there). Each run holds the GPU lock only while it runs.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
ENG=engines/mlx-lm-mtp
PY=engines/.venv-mtp/bin/python
LOCK="${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}"
N=${N_PROMPTS:-48}; L=${MAX_TOKENS:-256}; OUT=${OUT:-results/r2x}
run() { # label commit patch-sed-expr spec setting seed
  local label=$1 commit=$2 expr=$3 spec=$4 S=$5 seed=$6
  (cd $ENG && git checkout -q -f $commit) || return 1
  if [ -n "$expr" ]; then
    sed -i '' "$expr" $ENG/mlx_lm/generate.py
    n=$(grep -c "verify_lp, True" $ENG/mlx_lm/generate.py); echo "[$label] reporting patch applied: $n site(s)"
  fi
  GPU_LOCK_WAIT_MAX=${GPU_LOCK_WAIT_MAX:-10800} $LOCK "R2x $label $S spec-$spec seed $seed, bf16 Qwen3.5-4B, ~12 min, ~10 GB" \
    nice -n 5 $PY -I experiments/r2_mlx_mtp.py --label $label --spec $spec --setting $S --seed $seed \
    --n-prompts $N --max-tokens $L --out $OUT 2>&1 | grep -vE "Warning|z = -_Ak"
  (cd $ENG && git checkout -q -f c4963b1b9e)
}
TH=qwen_think_T0.6_p0.95_k20
if [ "${SMOKE:-0}" = 1 ]; then
  run D_head c4963b1b9e "" on $TH 104; exit 0
fi
run D_head      c4963b1b9e ""                                                                       off $TH 105
run D_head      c4963b1b9e ""                                                                       on  $TH 104
run C_preLpAcc  96093cfae0 's/yield draft_tok_id, draft_lp, True/yield draft_tok_id, verify_lp, True/' on  $TH 103
run B_preResid  673b3044da 's/yield draft_tok_id, draft_lp, True/yield draft_tok_id, verify_lp, True/' on  $TH 102
run A_initial   1ff33d56a0 's/yield draft_tok, draft_lp, True/yield draft_tok, verify_lp, True/'       on  $TH 101
run D_head      c4963b1b9e ""                                                                       off T1.0_plain 205
echo "R2x done"
