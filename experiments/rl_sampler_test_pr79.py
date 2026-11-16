"""Independent review of raceline PR #79 (sampler tail fix): the same exact L1 test as rl_sampler_test.py, but calling
the PR's real `_sample` (and its C top-k) from the worktree, and with the probability mass beyond the stored top-4096
put back as filler tokens, so the input vector and the declared full-vocabulary distribution agree exactly.

Run: python -I rl_sampler_test_pr79.py [n_seeds] [out.json]   (CPU only; reads results/rl_logits.npz)
Point RACELINE_SRC at any raceline checkout (e.g. an archive of v0.8.0) to test that version on the same input.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import paths  # noqa: E402
if not (paths.RACELINE_SRC and paths.RACELINE_LIB):
    raise SystemExit("set RACELINE_SRC (raceline checkout) and RACELINE_LIB (libraceline.dylib)")
sys.path.insert(0, paths.RACELINE_SRC)

import raceline.engine as E  # noqa: E402  (PR #79 worktree)
from raceline.backends import Sampling  # noqa: E402
from lossless import pit  # noqa: E402
from rl_sampler_test import claimed  # noqa: E402

V = 248320
N_FILL = 200_000


def build_full(ids, vals, lse, rng_ids):
    """Top-4096 exact logits + N_FILL equal filler logits carrying the true mass beyond rank 4096 (at T=1)."""
    head_lse = float(np.logaddexp.reduce(vals.astype(np.float64)))
    m = max(0.0, 1.0 - np.exp(head_lse - lse))            # true mass beyond rank 4096
    full = np.full(V + N_FILL, -np.inf, dtype=np.float32)   # filler ids live beyond the real vocabulary
    full[ids] = vals
    fill_val = None
    if m > 0:
        fill_val = lse + np.log(m) - np.log(N_FILL)
        fill_val = min(fill_val, float(vals[-1]) - 1e-3)     # fillers must rank below the stored head
        full[V:] = fill_val
    return full, m, fill_val


def main(n_seeds: int = 20, out_name: str = "rl_sampler_test_pr79.json"):
    d = np.load(ROOT / "results" / "rl_logits.npz")
    temps = list(d["temps"])
    settings = {
        "T1.0_k0_p1.0 (declared: full vocabulary)": (1.0, 0, 1.0, 0.0),
        "T1.0_k0_p0.95": (1.0, 0, 0.95, 0.0),
        "think T0.6_k20_p0.95": (0.6, 20, 0.95, 0.0),
        "nothink T0.7_k20_p0.8": (0.7, 20, 0.8, 0.0),
        "T1.0_k0_p1.0_minp0.05": (1.0, 0, 1.0, 0.05),
    }
    rng = np.random.default_rng(0)
    res = {}
    for name, (T, k, tp, mp) in settings.items():
        zs, s_, e_, v_, viol, beyond1024, beyond4096, exp1024, exp4096 = [], [], [], [], 0, 0, 0, 0.0, 0.0
        for i in range(d["ids"].shape[0]):
            ids, vals, lse1 = d["ids"][i], d["logits"][i], float(d["lse"][i])
            full, m4096, fill_val = build_full(ids, vals, lse1, rng)
            # declared distribution over the stored head (+ the filler bucket as one tail category)
            if T == 1.0:
                lse_T = lse1
            else:
                j = temps.index(T)
                top = np.logaddexp.reduce(vals[:1024].astype(np.float64) / T)
                lse_T = float(top - np.log1p(-d["tail1024"][i, j]))
            q, qtail = claimed(vals, lse_T, T, k, tp, mp)
            gstart = np.r_[0, np.nonzero(np.diff(vals) != 0)[0] + 1]
            gid = np.repeat(np.arange(gstart.size), np.diff(np.r_[gstart, vals.size]))
            gq = np.bincount(gid[: len(q)], weights=q, minlength=gstart.size)
            gcum = np.concatenate([[0.0], np.cumsum(gq)])
            ent_terms = np.concatenate([gq, [qtail]]) if qtail > 0 else gq
            logq = np.log(np.maximum(ent_terms, 1e-300))
            ent = float(-(ent_terms * logq).sum()); m2 = float((ent_terms * logq ** 2).sum())
            pos_of = {int(t): jj for jj, t in enumerate(ids)}
            if k == 0 and tp >= 1.0 and mp == 0:
                head1024 = np.logaddexp.reduce(vals[:1024].astype(np.float64) / T)
                exp1024 += n_seeds * float(1.0 - np.exp(head1024 - lse_T))
                exp4096 += n_seeds * m4096
            for sd in range(n_seeds):
                y = E._sample(full, Sampling(temperature=T, top_p=tp, top_k=k, min_p=mp), [], sd,
                              int(d["pos"][i]) + 100000 * i)
                jj = pos_of.get(y)
                if jj is None:                       # a filler (beyond rank 4096)
                    beyond1024 += 1; beyond4096 += 1
                    if qtail > 0:
                        zs.append(gcum[-1] + rng.random() * qtail); s_.append(-np.log(qtail))
                    else:
                        viol += 1; zs.append(1.0); s_.append(700.0)
                else:
                    if jj >= 1024:
                        beyond1024 += 1
                    g = gid[jj]
                    if gq[g] > 0:
                        zs.append(gcum[g] + rng.random() * gq[g]); s_.append(-np.log(gq[g]))
                    else:
                        viol += 1; zs.append(1.0); s_.append(700.0)
                e_.append(ent); v_.append(max(0.0, m2 - ent ** 2))
        rep = pit.uniformity_tests(np.array(zs), surprisal=(np.array(s_), np.array(e_), np.array(v_)))
        out = rep.as_dict() | {"reject_bonf_0.01": bool(pit.bonferroni_reject(rep, 0.01)), "draws": len(zs),
                               "support_violations": viol, "draws_beyond_rank_1024": beyond1024,
                               "draws_beyond_rank_4096": beyond4096}
        if exp1024:
            out |= {"expected_beyond_1024": exp1024, "expected_beyond_4096": exp4096}
        res[name] = out
        print(name, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in out.items()
                                if kk in ("draws", "ks_p", "mean_z", "surprisal_z", "reject_bonf_0.01", "support_violations",
                                          "draws_beyond_rank_1024", "expected_beyond_1024", "draws_beyond_rank_4096",
                                          "expected_beyond_4096")}), flush=True)
    (ROOT / "results" / out_name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, sys.argv[2] if len(sys.argv) > 2 else "rl_sampler_test_pr79.json")
