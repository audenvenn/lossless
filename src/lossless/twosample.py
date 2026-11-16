"""Exact two-sample tests between two engine arms (e.g. speculation on vs off).

Unit = one generated sequence. Under H0 (both arms sample from the same sequence distribution), arm labels are
exchangeable across sequences from the same prompt, so a permutation of labels WITHIN each prompt gives an exact
test for any statistic. Per-token quantities (PIT under a fixed reference model, surprisal, ...) are computed by
a fixed scoring function shared by both arms; the reference need not match the engine's numerics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SeqStats:
    prompt: np.ndarray   # (S,) prompt id per sequence
    arm: np.ndarray      # (S,) 0/1
    tokens: list         # list of per-token arrays per sequence (e.g. z_ref values)


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ks_2samp(a, b).statistic)


def _stat_fns():
    return {
        "mean": lambda a, b: float(np.mean(a) - np.mean(b)),
        "ks": _ks_distance,
        # weight the upper tail (surprising tokens) — sensitive to "too flat" sampling
        "upper_tail_frac": lambda a, b: float(np.mean(a > 0.9) - np.mean(b > 0.9)),
        "lower_tail_frac": lambda a, b: float(np.mean(a < 0.1) - np.mean(b < 0.1)),
    }


def permutation_test(prompt: np.ndarray, arm: np.ndarray, per_seq_values: list, n_perm: int = 2000,
                     seed: int = 0, strata: np.ndarray | None = None) -> dict:
    """Stratified permutation test of arm 1 vs arm 0 on pooled per-token values.

    per_seq_values[i] is a 1-D array of token-level values for sequence i (e.g. reference PIT z).
    Labels are permuted within strata (default: prompt). Use strata = (prompt, sample index) when both arms
    generate their k-th sample of a prompt under the same server conditions (e.g. prompt-cache state), so that
    exchangeability holds even if those conditions matter.

    Returns observed statistics, two-sided permutation p-values per statistic, and a single multiplicity-adjusted
    p-value for the family: Westfall-Young single-step min-p over the statistics (exact under H0, accounts for
    their dependence), alongside the Bonferroni bound for reference. p-values include the observed labelling,
    so the smallest attainable value is 1/(n_perm+1).
    """
    rng = np.random.default_rng(seed)
    prompt = np.asarray(prompt)
    arm = np.asarray(arm).astype(int)
    strata = prompt if strata is None else np.asarray(strata)
    fns = _stat_fns()
    keys = list(fns)

    def pooled(lbl):
        a = np.concatenate([v for v, l in zip(per_seq_values, lbl) if l == 1]) if (lbl == 1).any() else np.array([])
        b = np.concatenate([v for v, l in zip(per_seq_values, lbl) if l == 0]) if (lbl == 0).any() else np.array([])
        return a, b

    a, b = pooled(arm)
    obs = {k: f(a, b) for k, f in fns.items()}
    groups = [np.nonzero(strata == g)[0] for g in np.unique(strata)]
    # fast exact evaluation: only labels change between permutations, so use per-sequence sums and one pooled
    # sort order (identical values to the direct formulas in _stat_fns, which are kept for the observed labels)
    vals = [np.asarray(v, dtype=np.float64) for v in per_seq_values]
    n_s = np.array([v.size for v in vals], dtype=np.float64)
    sum_s = np.array([v.sum() for v in vals])
    hi_s = np.array([(v > 0.9).sum() for v in vals], dtype=np.float64)
    lo_s = np.array([(v < 0.1).sum() for v in vals], dtype=np.float64)
    allv = np.concatenate(vals)
    seq_of = np.repeat(np.arange(len(vals)), n_s.astype(int))
    o = np.argsort(allv, kind="stable")
    seq_sorted = seq_of[o]
    last = np.r_[allv[o][1:] != allv[o][:-1], True]
    ranks = np.arange(1, allv.size + 1)[last]

    def fast(lbl):
        la = lbl.astype(np.float64)
        na, nb = n_s @ la, n_s @ (1 - la)
        if na == 0 or nb == 0:
            return np.zeros(len(keys))
        ca = np.cumsum(la[seq_sorted])[last]
        ks = np.max(np.abs(ca / na - (ranks - ca) / nb))
        mean = sum_s @ la / na - sum_s @ (1 - la) / nb
        up = hi_s @ la / na - hi_s @ (1 - la) / nb
        dn = lo_s @ la / na - lo_s @ (1 - la) / nb
        out = {"mean": mean, "ks": ks, "upper_tail_frac": up, "lower_tail_frac": dn}
        return np.array([abs(out[k]) for k in keys])

    T = np.empty((n_perm + 1, len(keys)))
    T[0] = fast(arm)
    for i in range(1, n_perm + 1):
        lbl = arm.copy()
        for g in groups:
            lbl[g] = rng.permutation(lbl[g])
        T[i] = fast(lbl)
    # per-statistic p-value of every labelling (observed = row 0) within the permutation distribution
    P = np.empty_like(T)
    for j in range(len(keys)):
        col = T[:, j]
        srt = np.sort(col)
        P[:, j] = (col.size - np.searchsorted(srt, col - 1e-15, side="left")) / col.size
    pvals = {k: float(P[0, j]) for j, k in enumerate(keys)}
    minp = P.min(axis=1)
    p_wy = float(np.mean(minp <= minp[0] + 1e-15))
    return {"observed": obs, "p": pvals, "p_minp_wy": p_wy,
            "p_min_bonferroni": float(min(1.0, len(keys) * min(pvals.values()))),
            "n_seq": int(arm.size), "n_strata": len(groups), "n_tok_arm1": int(a.size), "n_tok_arm0": int(b.size),
            "n_perm": n_perm}
