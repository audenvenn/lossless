#!/bin/bash
# Drive R2 over historical commits of mlx-lm PR #990. Each run holds the GPU lock only while it runs.
set -u
cd "$(dirname "$0")/.."
. experiments/_env.sh
ENG=engines/mlx-lm-mtp
PY=engines/.venv-mtp/bin/python
LOCK="${GPU_LOCK_WRAPPER:-experiments/_nolock.sh}"
N=${N_PROMPTS:-32}; L=${MAX_TOKENS:-256}
SETS=${SETS:-"T1.0_plain qwen_think_T0.6_p0.95_k20"}
run() { # label commit patch-sed-expr spec
  local label=$1 commit=$2 expr=$3 spec=$4
  (cd $ENG && git checkout -q -f $commit) || return 1
  if [ -n "$expr" ]; then
    sed -i '' "$expr" $ENG/mlx_lm/generate.py
    n=$(grep -c "verify_lp, True" $ENG/mlx_lm/generate.py); echo "[$label] reporting patch applied: $n site(s)"
  fi
  for S in $SETS; do
    GPU_LOCK_WAIT_MAX=7200 $LOCK "R2 $label $S spec-$spec, bf16 Qwen3.5-4B, ~10 min, ~10 GB" \
      nice -n 5 $PY -I experiments/r2_mlx_mtp.py --label $label --spec $spec --setting $S \
      --n-prompts $N --max-tokens $L 2>&1 | grep -vE "Warning|z = -_Ak"
  done
  (cd $ENG && git checkout -q -f c4963b1b9e)
}
run D_head      c4963b1b9e ""                                                                       off
run D_head      c4963b1b9e ""                                                                       on
run A_initial   1ff33d56a0 's/yield draft_tok, draft_lp, True/yield draft_tok, verify_lp, True/'       on
run B_preResid  673b3044da 's/yield draft_tok_id, draft_lp, True/yield draft_tok_id, verify_lp, True/' on
run C_preLpAcc  96093cfae0 's/yield draft_tok_id, draft_lp, True/yield draft_tok_id, verify_lp, True/' on
echo "R2 done"
