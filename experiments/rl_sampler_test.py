"""Exact, server-free L1 test of raceline's numpy sampler (raceline 0.8.0, raceline/engine.py: `_uniform`, `_sample`,
copied verbatim below; only the C top-k helper `_top_k` is replaced by an equivalent numpy implementation).

For realistic logit vectors (results/rl_logits.npz: Qwen3.5-4B, top-4096 exact logits per position + full-vocabulary
normalizers), draw with many seeds and test the draws against the analytic distribution with the randomized PIT
(top-4096 + one tail bucket; exact by Lemma 3). Also counts draws beyond rank 1024, which raceline cannot produce
when top_k=0 ("1024: the tail carries no mass") while the declared distribution gives them mass `tail1024`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lossless import pit  # noqa: E402


@dataclass
class Sampling:  # raceline/backends.py
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None


def _top_k(logits: np.ndarray, k: int):
    """numpy stand-in for raceline's C rl_top_k: the k largest entries, descending."""
    k = min(k, logits.size)
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx], kind="stable")]
    return idx.astype(np.int64), logits[idx].astype(np.float32)


# ---- verbatim from raceline/engine.py (0.8.0), lines 590-662 ----
def _uniform(seed: int, position: int, ids: np.ndarray) -> np.ndarray:
    x = ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64((seed * 0x632BE59BD9B4E019 + position) % (1 << 64))
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def _sample(logits: np.ndarray, s: Sampling, recent: list, seed: int, position: int, with_probability: bool = False,
            tail=None):
    def ids(entries):
        if tail is None:
            return entries
        main, first = tail
        return np.where(entries < main, entries, entries - main + first)

    greedy = s.temperature <= 0
    k = (8 if with_probability else 1) if greedy else (s.top_k if s.top_k > 0 else 1024)   # 1024: the tail carries no mass
    top, z = _top_k(logits, k)
    if greedy:
        return int(ids(top[0]))
    logp = (z - z[0]).astype(np.float64) / s.temperature
    p = np.exp(logp)
    p /= p.sum()
    keep = len(top)
    if 0 < s.top_p < 1:
        keep = min(keep, int(np.searchsorted(np.cumsum(p), s.top_p)) + 1)
    if s.min_p > 0:
        keep = min(keep, int(np.count_nonzero(p >= s.min_p * p[0])))
    top, logp = ids(top[:keep]), logp[:keep]
    gumbel = -np.log(-np.log(_uniform(seed, position, top)))
    pick = int(np.argmax(logp + gumbel))
    return int(top[pick])
# ---- end verbatim (penalty branch omitted: no penalties are used here) ----


def claimed(vals: np.ndarray, lse_T: float, T: float, top_k: int, top_p: float, min_p: float):
    """Declared distribution over the stored top-K (descending) + tail bucket, HF-order semantics
    (temperature -> top-k -> top-p -> min-p); top_k=0 means no truncation (the full vocabulary)."""
    z = vals.astype(np.float64) / T
    if top_k > 0:
        z = z[:top_k]
        p = np.exp(z - np.logaddexp.reduce(z))
        tail = 0.0
    else:
        p = np.exp(z - lse_T)                      # full-vocabulary normalization
        tail = max(0.0, 1.0 - p.sum())
    if 0 < top_p < 1:
        cum = np.cumsum(p)
        keep = int(np.searchsorted(cum, top_p * (p.sum() + tail))) + 1
        p, tail = p[:keep], 0.0
    if min_p > 0:
        thr = min_p * p[0]
        p = np.where(p >= thr, p, 0.0)
        # the tail (beyond the stored head) lies below the smallest stored entry; it survives min-p only if that does
        if tail > 0 and np.exp((float(vals[-1]) - float(vals[0])) / T) < min_p:
            tail = 0.0
    tot = p.sum() + tail
    return p / tot, tail / tot


def main(n_seeds: int = 40):
    d = np.load(ROOT / "results" / "rl_logits.npz")
    V = 248320
    temps = list(d["temps"])
    settings = {
        "T1.0_k0_p1.0 (declared: full vocabulary)": (1.0, 0, 1.0, 0.0),
        "T1.0_k0_p0.95": (1.0, 0, 0.95, 0.0),
        "think T0.6_k20_p0.95": (0.6, 20, 0.95, 0.0),
        "nothink T0.7_k20_p0.8": (0.7, 20, 0.8, 0.0),
    }
    rng = np.random.default_rng(0)
    res = {}
    for name, (T, k, tp, mp) in settings.items():
        zs, beyond1024, exp_beyond, s_, e_, v_, support_viol = [], 0, 0.0, [], [], [], 0
        for i in range(d["ids"].shape[0]):
            ids, vals = d["ids"][i], d["logits"][i]
            if T == 1.0:
                lse_T = float(d["lse"][i])
            else:
                j = temps.index(T) if T in temps else None
                top = np.logaddexp.reduce(vals[:1024].astype(np.float64) / T)
                lse_T = float(top - np.log1p(-d["tail1024"][i, j])) if j is not None else np.nan
            q, qtail = claimed(vals, lse_T, T, k, tp, mp)
            full = np.full(V, -np.inf, dtype=np.float32)
            full[ids] = vals
            rank_of = {int(t): r for r, t in enumerate(ids[: max(len(q), 1024 + 1)])}
            cum = np.concatenate([[0.0], np.cumsum(q)])
            # entropy of the coarsened (value-group + tail) variable, matching the PIT categories
            _g0 = np.r_[0, np.nonzero(np.diff(vals) != 0)[0] + 1]
            _gid = np.repeat(np.arange(_g0.size), np.diff(np.r_[_g0, vals.size]))
            _gq = np.bincount(_gid[: len(q)], weights=q, minlength=_g0.size)
            ent_terms = np.concatenate([_gq, [qtail]]) if qtail > 0 else _gq
            logq = np.log(np.maximum(ent_terms, 1e-300))
            ent = float(-(ent_terms * logq).sum()); m2 = float((ent_terms * logq ** 2).sum())
            if k == 0 and tp >= 1.0:
                exp_beyond += n_seeds * float(1.0 - np.exp(np.logaddexp.reduce(vals[:1024].astype(np.float64) / T) - lse_T))
            # tie-robust categories (Lemma 3): tokens with identical logits form one category, since their
            # probabilities are identical and which tied member a top-p boundary keeps is implementation-defined
            gstart = np.r_[0, np.nonzero(np.diff(vals) != 0)[0] + 1]           # run starts in descending order
            gid = np.repeat(np.arange(gstart.size), np.diff(np.r_[gstart, vals.size]))
            gq = np.bincount(gid[: len(q)], weights=q, minlength=gstart.size)   # claimed mass per value group
            gcum = np.concatenate([[0.0], np.cumsum(gq)])
            pos_of = {int(t): j for j, t in enumerate(ids)}
            for sd in range(n_seeds):
                y = _sample(full, Sampling(temperature=T, top_p=tp, top_k=k, min_p=mp), [], sd, int(d["pos"][i]) + 100000 * i)
                j = pos_of.get(y)
                if j is not None and j >= 1024:
                    beyond1024 += 1
                g = gid[j] if j is not None else None
                if g is not None and gq[g] > 0:
                    zs.append(gcum[g] + rng.random() * gq[g]); s_.append(-np.log(gq[g]))
                else:  # outside the stored head or outside the claimed support
                    zs.append(gcum[-1] + rng.random() * qtail if qtail > 0 else 1.0)
                    s_.append(-np.log(qtail) if qtail > 0 else 700.0)
                    if qtail == 0:
                        support_viol += 1
                e_.append(ent); v_.append(max(0.0, m2 - ent ** 2))
        rep = pit.uniformity_tests(np.array(zs), surprisal=(np.array(s_), np.array(e_), np.array(v_)))
        out = rep.as_dict() | {"reject_bonf_0.01": bool(pit.bonferroni_reject(rep, 0.01)), "draws": len(zs),
                               "draws_beyond_rank_1024": beyond1024, "support_violations": support_viol}
        if exp_beyond:
            out["expected_beyond_1024_under_declared"] = exp_beyond
            out["poisson_p_observing_0"] = float(np.exp(-exp_beyond))
        res[name] = out
        print(name, json.dumps({k2: (round(v, 6) if isinstance(v, float) else v) for k2, v in out.items()
                                if k2 in ("draws", "ks_p", "mean_z", "surprisal_z", "reject_bonf_0.01",
                                          "draws_beyond_rank_1024", "support_violations", "expected_beyond_1024_under_declared",
                                          "poisson_p_observing_0")}), flush=True)
    t = d["tail1024"]
    res["tail_mass_beyond_1024"] = {f"T={T}": {"mean": float(t[:, j].mean()), "p50": float(np.median(t[:, j])),
                                               "p90": float(np.quantile(t[:, j], 0.9)),
                                               "p99": float(np.quantile(t[:, j], 0.99)), "max": float(t[:, j].max())}
                                    for j, T in enumerate(temps)}
    (ROOT / "results" / "rl_sampler_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["tail_mass_beyond_1024"], indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
