"""Toy: L1 on natural sequences vs fixed-prefix chi-square counting (the practice in vLLM's tests and llama.cpp
#29975) at matched token budgets, deployment setting. Also reports the toy's draft acceptance rate.

Fixed-prefix test: draw the first token after a fixed context N times with the speculative sampler and compare the
counts to the declared distribution P[context] by a chi-square goodness-of-fit test (cells with expected count < 5
merged). Variants: one random typical context with all N draws, or ten contexts with N/10 each (statistics summed).
Usage: python -I toy_baseline.py [reps]  -> results/toy_baseline.json (CPU only)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lossless import pit, toy  # noqa: E402

VARIANTS = ("correct", "resample_target", "accept_untransformed", "residual_untransformed", "bonus_from_draft",
            "argmax_on_reject", "target_kv_stale")
BUDGETS = (3000, 10000, 30000)


def chi2_stat(counts, p, min_exp=5.0):
    e = p * counts.sum()
    big = e >= min_exp
    o_all = np.r_[counts[big], counts[~big].sum()]
    e_all = np.r_[e[big], e[~big].sum()]
    keep = e_all > 0
    o_all, e_all = o_all[keep], e_all[keep]
    return float(((o_all - e_all) ** 2 / e_all).sum()), int(o_all.size - 1), int(counts[p == 0].sum())


def first_tokens(model, P, Q, P1, Q1, K, variant, ctx, n, rng):
    c1, c2, y, ch = toy.simulate_spec(model, P, Q, P1, Q1, K, 1, variant, rng, B=n, return_chain=True, start=ctx)
    first = np.full(n, -1)
    seen = np.zeros(n, bool)
    for t, c in zip(y, ch):                 # output order: accepted drafts by slot, then the final token
        if not seen[c]:
            first[c] = t; seen[c] = True
    return first


def main(reps=100, seed=0, K=3):
    model = toy.ToyLM.random(V=48, seed=seed)
    s = dict(temp=0.7, top_k=20, top_p=0.95)
    P, Q = model.tables(**s)
    P1, Q1 = model.tables(temp=1.0)
    V = model.V
    rng0 = np.random.default_rng(seed + 1)
    c1v, c2v, _ = toy.simulate_plain(P, 20000, rng0)
    acc = {}
    for name, stt in (("deploy", s), ("T1", dict(temp=1.0))):
        Pp, Qq = model.tables(**stt)
        d = {}
        toy.simulate_spec(model, Pp, Qq, P1, Q1, K, 50000, "correct", np.random.default_rng(3), stats=d)
        acc[name] = d["accepted"] / d["proposed"]
    print("toy draft acceptance rate:", acc, flush=True)
    res = {"acceptance_rate": acc, "reps": reps, "alpha": 0.01, "seed": seed, "chain_tokens": 256, "power": {}}
    for v in VARIANTS:
        res["power"][v] = {}
        for N in BUDGETS:
            hit = {"L1": 0, "chi2_1prefix": 0, "chi2_10prefix": 0}
            for r in range(reps):
                rng = np.random.default_rng(1000 * r + N + hash(v) % 997)
                # L1 on natural sequences (same battery as the paper, Bonferroni at 0.01)
                # realistic sequences of ~256 tokens (ceil(N/256) chains), as in the real runs. Truncate in CHAIN
                # order: the simulator emits round by round, and dropping a round's overshoot would select tokens by
                # draft slot and acceptance outcome (an invalid stratum, Lemma 2).
                a1, a2, y, ch = toy.simulate_spec(model, P, Q, P1, Q1, K, N, v, rng, B=int(np.ceil(N / 256)),
                                                  return_chain=True)
                o = np.argsort(ch, kind="stable")
                a1, a2, y = a1[o][:N], a2[o][:N], y[o][:N]
                z = toy.pit_values(P, a1, a2, y, rng)
                sur = toy.surprisal_from_tables(P, a1, a2, y)
                viol = int((P[a1, a2, y] <= 0).sum())
                rep = pit.uniformity_tests(z, surprisal=sur)
                hit["L1"] += int(viol > 0 or pit.bonferroni_reject(rep, 0.01))
                # fixed-prefix chi-square: typical contexts drawn from correct generations
                for name, m in (("chi2_1prefix", 1), ("chi2_10prefix", 10)):
                    idx = rng.integers(c1v.size, size=m)
                    tot, df, imp = 0.0, 0, 0
                    for i in idx:
                        ctx = (int(c1v[i]), int(c2v[i]))
                        f = first_tokens(model, P, Q, P1, Q1, K, v, ctx, N // m, rng)
                        counts = np.bincount(f, minlength=V).astype(float)
                        x, k, bad = chi2_stat(counts, P[ctx])
                        tot += x; df += k; imp += bad
                    p = st.chi2.sf(tot, max(df, 1)) if df > 0 else 1.0
                    hit[name] += int(imp > 0 or p < 0.01)
            res["power"][v][N] = {k: hit[k] / reps for k in hit}
            print(f"{v:24s} N={N:6d} " + "  ".join(f"{k} {hit[k]/reps:.2f}" for k in hit), flush=True)
    (ROOT / "results" / "toy_baseline.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
