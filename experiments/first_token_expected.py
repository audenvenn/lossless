"""Analytic expectation for the first-token minimal reproduction (first_token_support.py), from the reference model.

Distinct first tokens in n = 400 independent draws at T = 1: E = sum_v 1 - (1 - p_v)^n, plus the simulated
distribution of the count, for (a) the full vocabulary and (b) a top-32 cap. Reference: MLX 4-bit Qwen3.5-4B
(an engine with different weights gives a somewhat different p, so the full-vocabulary value is approximate).

Usage (under the GPU lock): python -I first_token_expected.py   -> results/first_token_expected.json
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
from first_token_support import PROMPT  # noqa: E402


def main(n=400, sims=20000):
    from mlx_lm import load
    model, tok = load(paths.QWEN_MLX4)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True,
                                  enable_thinking=False)
    lg = np.array(model(mx.array(ids)[None])[0, -1].astype(mx.float32), dtype=np.float64)
    p = np.exp(lg - np.logaddexp.reduce(lg))
    rng = np.random.default_rng(0)
    out = {"n_draws": n, "prompt": PROMPT}
    for name, q in (("full_vocabulary", p), ("top32_cap", None)):
        if q is None:
            top = np.argsort(-p)[:32]
            q = np.zeros_like(p); q[top] = p[top]; q /= q.sum()
        exp = float((1 - (1 - q) ** n).sum())
        cdf = np.cumsum(q); cdf /= cdf[-1]
        draws = np.searchsorted(cdf, rng.random((sims, n)), side="right")      # sims x n categorical draws
        srt = np.sort(draws, axis=1)
        counts = 1 + (np.diff(srt, axis=1) != 0).sum(axis=1)                    # distinct tokens per simulation
        out[name] = {"expected_distinct": exp, "sims": int(sims),
                     "sim_quantiles": {k: float(np.quantile(counts, k)) for k in (0.001, 0.005, 0.5, 0.995, 0.999)},
                     "p_at_most_32": float(np.mean(counts <= 32))}
        print(name, json.dumps(out[name]), flush=True)
    out["mass_beyond_rank32"] = float(np.sort(p)[::-1][32:].sum())
    (ROOT / "results" / "first_token_expected.json").write_text(json.dumps(out, indent=1))
    top = np.argsort(-p)[:4096]
    np.savez_compressed(ROOT / "results" / "first_token_reference_dist.npz", ids=top.astype(np.int32), p=p[top],
                        mass_beyond_4096=float(1 - p[top].sum()))


if __name__ == "__main__":
    main()
