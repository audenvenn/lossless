"""R3: per-token divergence hazard of greedy speculative decoding vs plain greedy decoding (Kaplan-Meier + geometric MLE).

A naive identity check reports "x/48 identical"; that number depends on output length. Here each prompt contributes
the positions it matched before its first divergence (an event) or, if never diverged, its full length (censored).
Hazard = events / positions at risk, with an exact Poisson 95% CI; rate ratios against the correct head D use the
exact conditional binomial test. CPU only; reads results/r3/*.seqs.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

R3 = Path(__file__).resolve().parents[1] / "results" / "r3"


def times(tag):
    ref = {s["prompt"]: s["tokens"] for s in json.loads((R3 / "D_head_greedy_specoff.seqs.json").read_text())["seqs"]}
    seqs = {s["prompt"]: s["tokens"] for s in json.loads((R3 / f"{tag}.seqs.json").read_text())["seqs"]}
    out = []
    for p, toks in seqs.items():
        r = ref[p]
        if toks == r:
            out.append((len(toks), 0))                  # censored: matched every position
        else:
            n = next((i for i, (x, y) in enumerate(zip(toks, r)) if x != y), min(len(toks), len(r)))
            out.append((n + 1, 1))                      # n matched positions, divergence at position n
    return np.array(out)


def km_median(tt):
    t, e = tt[:, 0], tt[:, 1]
    s, med = 1.0, None
    for u in np.unique(t[e == 1]):
        at_risk = np.sum(t >= u)
        s *= 1 - np.sum((t == u) & (e == 1)) / at_risk
        if med is None and s <= 0.5:
            med = int(u)
    return med, s


def main():
    res = {}
    base = times("D_head_greedy_specon")
    d0, n0 = int(base[:, 1].sum()), int(base[:, 0].sum())
    for tag in ("A_initial_greedy_specon", "B_preResid_greedy_specon", "C_preLpAcc_greedy_specon", "D_head_greedy_specon"):
        tt = times(tag)
        d, n = int(tt[:, 1].sum()), int(tt[:, 0].sum())
        lo = stats.chi2.ppf(0.025, 2 * d) / 2 / n if d else 0.0
        hi = stats.chi2.ppf(0.975, 2 * d + 2) / 2 / n
        med, s_end = km_median(tt)
        rr = (d / n) / (d0 / n0) if d0 else np.nan
        p = stats.binomtest(d, d + d0, n / (n + n0), alternative="greater").pvalue if tag != "D_head_greedy_specon" else np.nan
        res[tag] = {"prompts": len(tt), "events": d, "positions_at_risk": n, "hazard_per_token": d / n,
                    "hazard_ci95": [lo, hi], "km_median_tokens": med, "km_survival_at_end": s_end,
                    "rate_ratio_vs_D": rr, "p_rate_greater_than_D": p}
        print(f"{tag:26s} events {d:2d}/{len(tt)}  at risk {n:6d}  hazard {100*d/n:.2f}%/token "
              f"[{100*lo:.2f}, {100*hi:.2f}]  KM median {med}  ratio vs D {rr:5.1f}  p {p:.2g}")
    (R3 / "hazard.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
