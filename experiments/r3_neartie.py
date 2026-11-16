"""R3 near-tie analysis: are greedy divergences of the correct head D numerics (near-ties), unlike those of A/B/C?

For each prompt, one prefill of the plain greedy sequence (D-head, speculation off) with the bf16 model gives the
next-token logits at every position. At a speculative run's first divergence position n (common prefix r[:n]) we
record the top-2 logit gap, both tokens' ranks, and the logit difference between the plain token and the speculative
token. Baseline: the top-2 gap at every position of the plain sequences. A numerics-driven flip should sit at a
near-tie (small gap, the two tokens are the top-2); an algorithmic error need not.

Usage (under the GPU lock): python -I r3_neartie.py   -> results/r3/neartie.json
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
mx.set_cache_limit(1 << 30)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import paths  # noqa: E402

R3 = ROOT / "results" / "r3"


def main():
    from mlx_lm import load
    model, _ = load(paths.QWEN_BF16)
    ref = {s["prompt"]: s for s in json.loads((R3 / "D_head_greedy_specoff.seqs.json").read_text())["seqs"]}
    runs = {t: {s["prompt"]: s["tokens"] for s in json.loads((R3 / f"{t}.seqs.json").read_text())["seqs"]}
            for t in ("A_initial_greedy_specon", "B_preResid_greedy_specon", "C_preLpAcc_greedy_specon",
                      "D_head_greedy_specon")}
    base_gaps, out = [], {t: [] for t in runs}
    for p, s in sorted(ref.items()):
        pid, r = s["prompt_ids"], s["tokens"]
        logits = model(mx.array(list(pid) + list(r))[None])[0, len(pid) - 1: len(pid) - 1 + len(r)]
        lg = np.array(logits.astype(mx.float32))
        mx.clear_cache()
        top2 = np.sort(lg, axis=1)[:, -2:]
        gaps = top2[:, 1] - top2[:, 0]
        base_gaps.extend(gaps.tolist())
        for t, seqs in runs.items():
            toks = seqs.get(p)
            if toks is None or toks == r:
                continue
            n = next((i for i, (x, y) in enumerate(zip(toks, r)) if x != y), None)
            if n is None or n >= len(r):
                continue                                   # divergence by length only (EOS), no token at n
            row = lg[n]
            order = np.argsort(-row)
            rank = {int(v): int(k) for k, v in enumerate(order[:1000])}
            out[t].append({"prompt": p, "pos": n, "top2_gap": float(gaps[n]), "plain_tok": int(r[n]),
                           "spec_tok": int(toks[n]), "rank_plain": rank.get(int(r[n]), 1000),
                           "rank_spec": rank.get(int(toks[n]), 1000),
                           "logit_plain_minus_spec": float(row[r[n]] - row[toks[n]])})
    base = np.array(base_gaps)
    summ = {"baseline_positions": int(base.size),
            "baseline_gap_quantiles": {q: float(np.quantile(base, q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5)}}
    for t, rows in out.items():
        g = np.array([x["top2_gap"] for x in rows])
        pct = np.array([float(np.mean(base <= x)) for x in g]) if g.size else np.array([])
        summ[t] = {"divergences": len(rows), "median_gap": float(np.median(g)) if g.size else None,
                   "median_baseline_percentile": float(np.median(pct)) if pct.size else None,
                   "frac_spec_token_is_runner_up": float(np.mean([x["rank_spec"] == 1 and x["rank_plain"] == 0
                                                                   for x in rows])) if rows else None,
                   "frac_gap_below_baseline_p05": float(np.mean(g <= summ["baseline_gap_quantiles"][0.05]))
                   if g.size else None}
        print(t, json.dumps(summ[t]), flush=True)
    (R3 / "neartie.json").write_text(json.dumps({"summary": summ, "rows": out}, indent=1))


if __name__ == "__main__":
    main()
