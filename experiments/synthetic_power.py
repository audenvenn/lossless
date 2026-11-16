"""Power of the PIT battery per bug class on a toy order-2 LM (exact conditionals).

For each (setting, variant): R replications; each simulates N_max emitted tokens and evaluates the test
battery on nested prefixes N in NS. Reports rejection rate at alpha=0.01 (Bonferroni over the battery)
and, for sampler-semantics variants, the exact mean per-token TV between true and claimed distributions.

Usage: python -I experiments/synthetic_power.py [--reps 40] [--out results/synthetic_power.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lossless import pit, samplers, toy  # noqa: E402

NS = [300, 1000, 3000, 10000, 30000, 100000]
SETTINGS = {
    "T1.0": dict(temp=1.0, top_k=0, top_p=1.0, min_p=0.0),
    "T0.7_k20_p0.95": dict(temp=0.7, top_k=20, top_p=0.95, min_p=0.0),
}
SPEC_VARIANTS = list(toy.VARIANTS)
SAMPLER_VARIANTS = ["temp_plus5pct", "temp_last_vs_first", "silent_min_p_0.05", "top_p_exclusive"]


def mask_top_p_exclusive(z, p):
    probs = samplers.softmax(z)
    idx = np.argsort(-z, kind="stable")
    cum = np.cumsum(probs[idx])
    n_keep = max(int(np.searchsorted(cum, p - 1e-12, side="left")), 1)  # drop the crossing token
    out = np.full_like(z, -np.inf)
    out[idx[:n_keep]] = z[idx[:n_keep]]
    return out


def true_tables(model, variant, s):
    """The distribution an engine with this sampler bug actually samples from (claimed = hf semantics)."""
    V = model.V
    P = np.empty((V, V, V))
    for i in range(V):
        for j in range(V):
            z = model.target_logits[i, j]
            if variant == "temp_plus5pct":
                P[i, j] = samplers.transform(z, s["temp"] * 1.05, s["top_k"], s["top_p"], s["min_p"], "hf")
            elif variant == "temp_last_vs_first":
                P[i, j] = samplers.transform(z, s["temp"], s["top_k"], s["top_p"], s["min_p"], "llamacpp")
            elif variant == "silent_min_p_0.05":
                P[i, j] = samplers.transform(z, s["temp"], s["top_k"], s["top_p"], 0.05, "hf")
            elif variant == "top_p_exclusive":
                zz = samplers.mask_top_k(z / s["temp"], s["top_k"])
                zz = mask_top_p_exclusive(zz, s["top_p"]) if s["top_p"] < 1 else zz
                P[i, j] = samplers.softmax(zz)
            else:
                raise ValueError(variant)
    return P


def run(reps: int, out: Path, V: int, K: int, seed: int):
    t0 = time.time()
    model = toy.ToyLM.random(V=V, seed=seed)
    results = {"NS": NS, "V": V, "K": K, "reps": reps, "alpha": 0.01, "settings": {}}
    for sname, s in SETTINGS.items():
        P, Q = model.tables(**s)
        P1, Q1 = model.tables(temp=1.0)
        # stationary-ish context weights for exact TV: use contexts visited by correct sampling
        rng0 = np.random.default_rng(seed + 99)
        c1, c2, _ = toy.simulate_plain(P, 50000, rng0)
        sres = {}
        variants = [("spec", v) for v in SPEC_VARIANTS] + [("sampler", v) for v in SAMPLER_VARIANTS]
        for kind, v in variants:
            if kind == "sampler":
                Ptrue = true_tables(model, v, s)
                tvs = 0.5 * np.abs(Ptrue[c1, c2] - P[c1, c2]).sum(axis=1)
                eff = {"mean_tv": float(tvs.mean()), "p90_tv": float(np.quantile(tvs, 0.9))}
            else:
                eff = {}
            rej = np.zeros(len(NS))
            rej_tests = {k: np.zeros(len(NS)) for k in ("ks", "cvm", "chi2", "meanz", "surprisal")}
            for r in range(reps):
                rng = np.random.default_rng(seed * 1000 + r)
                if kind == "spec":
                    c1r, c2r, y = toy.simulate_spec(model, P, Q, P1, Q1, K, NS[-1], v, rng)
                else:
                    c1r, c2r, y = toy.simulate_plain(Ptrue, NS[-1], rng)
                perm = rng.permutation(y.size)[: NS[-1]]  # mix chains/rounds before taking prefixes
                c1r, c2r, y = c1r[perm], c2r[perm], y[perm]
                z = toy.pit_values(P, c1r, c2r, y, rng)
                sur = toy.surprisal_from_tables(P, c1r, c2r, y)
                for ni, n in enumerate(NS):
                    rep = pit.uniformity_tests(z[:n], surprisal=tuple(a[:n] for a in sur))
                    a5 = 0.01 / 5
                    rej[ni] += pit.bonferroni_reject(rep, 0.01)
                    rej_tests["ks"][ni] += rep.ks_p < a5
                    rej_tests["cvm"][ni] += rep.cvm_p < a5
                    rej_tests["chi2"][ni] += rep.chi2_p < a5
                    rej_tests["meanz"][ni] += rep.mean_z_p < a5
                    rej_tests["surprisal"][ni] += (rep.surprisal_p or 1) < a5
            sres[v] = {"kind": kind, **eff, "power": (rej / reps).tolist(),
                       "power_by_test": {k: (x / reps).tolist() for k, x in rej_tests.items()}}
            print(f"[{time.time()-t0:7.1f}s] {sname:16s} {v:24s} power={np.round(rej/reps,2).tolist()} {eff}",
                  flush=True)
        results["settings"][sname] = sres
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--V", type=int, default=48)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "synthetic_power.json")
    a = ap.parse_args()
    run(a.reps, a.out, a.V, a.K, a.seed)
