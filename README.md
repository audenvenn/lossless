# lossless

**Is your speculative decoding actually lossless? Is your sampler sampling what it claims?**

Speculative decoding and multi-token prediction (MTP) are now on by default in local inference engines. The engines document claims such as:
- "acceptance is exact rejection sampling, correct at any temperature";
- "bit-for-bit what decoding one token at a time produces".

Such claims are usually checked by comparing greedy outputs.

That check has two problems:
- **It misses bug classes.** When temperature 0 is routed to exact-match verification, a wrong acceptance ratio or residual cannot change greedy output. Truncation bugs only appear at the setting that exercises them.
- **It fails correct implementations.** Batched verification rounds differently from one-token decoding, so near-ties flip. In the history we studied, the correct implementation diverged from plain greedy decoding at 0.6% per token, which is 79% of 256-token outputs.

`lossless` provides statistically exact tests that work at the sampling settings you actually ship.

**What we did *not* find:** any speculative-decoding error in a shipped engine.
- The speculative bugs below are in intermediate commits of an unmerged pull request; its author fixed all of them before our study.
- Both sampler deviations in shipped engines were found by reading source code, then confirmed and measured with these tests.

## Tests

| Test | Needs | Detects |
|---|---|---|
| **L1: randomized PIT** | per-token distributions: full logits, OpenAI-style `top_logprobs`, or llama.cpp `n_probs` (pre-sampling) | emitted tokens that do not follow the declared distribution: wrong residual or acceptance ratio, argmax on rejection, silent truncation, wrong temperature or order of operations |
| **L3: exact two-sample test** | two arms (e.g. speculation on vs off), scored by any fixed reference model | distribution changes between arms, including state bugs (un-rolled-back KV or recurrent state) that L1 cannot see |
| **Identity** | two same-seed reply files | whether bit-for-bit claims hold, and where outputs first diverge |

**Why L1 is exact.** If each token is drawn from the declared distribution, its randomized PIT values are iid Uniform(0,1), even though the distribution adapts to the generated prefix. Exactness survives three operations:
- selecting positions by any rule fixed before the token is drawn (optional skipping);
- merging tied tokens or the unreported tail into one category;
- skipping positions where a top-K report does not determine the declared distribution.

Engines differ in how they keep ties at a top-k/top-p boundary ("count" vs "all"), so `lossless` takes that rule into account. Positions that can't be computed exactly are reported as screened, never guessed. After screening, a single emitted token with zero declared probability rejects. See `paper/` for statements and proofs.

**L3** permutes arm labels within (prompt, sample) strata and reports a Westfall–Young family p-value over four statistics. It is calibrated (about 1% false rejections at α = 0.01 in simulation). At audit sizes (~12k tokens per arm) it reliably detects state errors and large shifts, **not** subtle acceptance errors: use L1 for those.

**Exactness conditions.**
- L1 is exact when the declared distribution is fixed before each token is drawn: fixed-shape or batch-invariant kernels.
- Screening is exact when its tolerance bounds the engine's own arithmetic.
- At near-deterministic positions the PIT is mostly the tester's own coin. Decisions should rest on support violations and the surprisal statistic. `lossless` keeps the coin fixed at generation time and can report derandomized p-values and an entropy-screened variant.

**Complementary to first-token counting.** In simulation with realistic sequence lengths, counting the first token after ten fixed prompts far exceeds L1 for two first-position errors: resample-from-target and an untransformed residual. L1 needs ~100k tokens for those.
- First-token counting is blind to errors that act later: bonus tokens, history-dependent state.
- L1 catches those, and most acceptance and transform errors, within a few thousand tokens, largely through impossible tokens.
- Use both.

## What it found (Qwen3.5-4B, Apple silicon, November 2026)

### Natural experiment: the history of mlx-lm MTP pull request #990

| Commit | What was wrong | Caught by |
|---|---|---|
| A | recurrent state not rolled back after a rejected draft (listed by its author as a known limitation and removed 24 minutes later); exact-match verification | L2 and L3, not L1 |
| B | rejection path resamples from the target; acceptance uses untransformed log-probabilities | L1 at Qwen's thinking setting |
| C | acceptance and residual both use untransformed log-probabilities | L1 at the thinking setting, and L3 |
| D | (fixed head) | passes every test |

- **A.** Prefill TV is 30–50× the numerical floor; outputs are 35% shorter at T = 1.
- **B.** 15 impossible tokens, all accepted drafts.
- **C.** 40 impossible tokens: 11 accepted drafts and 29 corrected tokens.
- **Greedy identity needs a baseline.** Divergence hazards are 0.6%/token for D vs 2.7–5.5% for A–C, and all of D's divergences sit at logit near-ties.

### Engine audit (speculation on vs off)

| Engine | L3 (Westfall–Young p) | Same-seed identity |
|---|---|---|
| kapitsa 0.2.36 | 0.12–0.24 | 81–88% byte-identical |
| MTPLX 2.12.3 | 0.63–0.79 | not seed-coupled (by design) |
| raceline 0.8.0/0.8.1 (developed by an author) | 0.19–0.91 | 128/128 byte-identical |
| llama.cpp 0.9.0 | 0.58 / 0.033 (A/A cell 0.034: seed sets, not speculation) | 83–100% byte-identical |

These results show no evidence of a speculation effect. For raceline, and for llama.cpp at the thinking setting, speculation is bit-identical, so L3 adds nothing beyond the identity check.

As a positive control under the same conditions (one run each), L3 detects commits A and C at the thinking setting, but not B. So the audit would likely catch state errors of A's size, not subtle acceptance errors of B's size.

### Samplers vs their declared distribution
Both deviations were found by reading source code and then confirmed with targeted exact tests.
- **raceline 0.8.0** kept only the top 1024 tokens when top_k = 0: 0 draws beyond rank 1024 against 41.5 expected (p ≈ 10⁻¹⁸).
  - The omnibus L1 battery does not catch this; the targeted count does.
  - **Fixed in 0.8.1** with an exact lazy-tail Gumbel construction (39 vs 41.5); speculation stays bit-identical.
- **kapitsa** clamps top_k to 32 whenever top_k is 0 or above 32, which is undocumented. It removes 1.1% (non-thinking text) or 0.48% (thinking text) of the probability mass per token on average at T = 1.
  - Minimal reproduction: 32 distinct first tokens over 400 seeds, where the full distribution gives about 121.
  - Reported upstream (nace-ai/kapitsa#1286).
- **llama.cpp**, tested with its pre-sampling log-probs against its declared order of operations, passes L1 at both settings (14,354 and 16,376 tokens tested). Under draft-MTP, its API reports a placeholder p = 1 for 99.4% of tokens; this is a known upstream issue (#27972).
- **mlx-lm's earlier top-p** (used in the PR) accumulates in bf16. It differs from exact semantics at about 3% of positions (mean TV 4·10⁻⁵): small but real, and present with speculation on and off alike.

### Sampler semantics differ between engines
The same model-card parameters define different distributions:
- temperature first: HF, vLLM, raceline, MTPLX;
- temperature last: llama.cpp, mlx-lm, kapitsa.

At Qwen's non-thinking settings (T 0.7, top-p 0.8, top-k 20), 2.2% of the tokens mlx-lm emits are impossible under HF semantics.

## Paper

*Exact Conformance Tests for Speculative Decoding and Samplers in LLM Inference Engines*, Auden Venn (Independent Researcher), November 2026. Source in `paper/` (`main.tex`; figures from `paper/make_figures.py`). arXiv link to follow.

## Usage

```sh
pip install -e .                    # numpy, scipy; transformers for `identity`
lossless simulate --reps 40         # power per bug class on a toy LM with exact conditionals
# L1 from top-logprob reports (pre-transform values; name the engine's order of operations)
lossless l1 replies.jsonl --temp 0.6 --top-p 0.95 --top-k 20 --semantics llamacpp
lossless l3 arm_on.npz arm_off.npz  # per-sequence fixed-reference scores (experiments/score_arms.py)
lossless identity on.jsonl off.jsonl --tokenizer Qwen/Qwen3.5-4B
```

- `--semantics` names the engine's order of operations: `hf`, `vllm`, `llamacpp`, `mlxlm` or `mlxlm_legacy`.
- For engines that keep every token tied with the k-th value (HF, vLLM), request at least top_k + 1 log-probs.
- `--post-transform` tests only the final draw when the server reports post-sampling probabilities. It cannot see a transform bug such as a silent cap.

## Reproducing the paper
- Model and engine paths come from environment variables (see `experiments/paths.py`). The `run_*.sh` scripts source a local `paths.local.env` and set `HF_HUB_OFFLINE=1`.
- GPU jobs call `$GPU_LOCK_WRAPPER "description" command...`. The default, `experiments/_nolock.sh`, just runs the command; on a shared machine, point it at your own lock script.
- Main scripts:
  - `r2_mlx_mtp.py` + `run_r2x.sh`: natural experiment, white-box capture.
  - `r2x_analyze.py`, `l2_matched.py`, `r3_hazard.py`, `r3_neartie.py`.
  - `score_arms.py` + `l3_compare.py`: L3.
  - `toy_l3.py`.
  - `rl_sampler_test_pr79.py`: raceline sampler, any version.
- Result summaries are in `results/`. The figures come from `paper/make_figures.py`.

## Disclosure
- raceline is developed by the author (Auden Venn); it was tested with the same scripts, and its deviation was fixed before publication.
- This work was done with substantial help from an AI assistant (Anthropic's Claude), which wrote most of the code and text under the author's direction. The AI is not an author; the author takes full responsibility for the content.
- License: Apache-2.0.
