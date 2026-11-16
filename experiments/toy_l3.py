"""Toy L3: calibration under H0 and power of the exact two-sample test, scored by an *approximate* fixed reference.

Setup mirrors the real natural experiment: per replication, arm 1 = speculative decoding (a variant), arm 0 = plain
sampling, B = 48 chains of ~256 tokens each (one chain per "prompt"; chain i of arm 1 is paired with chain i of
arm 0, labels permuted within pairs). Every token is scored by a reference model whose logits differ from the true
model by Gaussian noise (sigma; a stand-in for a different quantization), at T = 1 without truncation. Statistic:
Westfall-Young min-p over the four L3 statistics. Also a 'temperature shift' family where arm 1 samples at
T(1+x): small, exactly known per-token TV, to map L3's minimum detectable effect.

Usage: python -I toy_l3.py [reps] [out.json]   (CPU only)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lossless import toy, twosample  # noqa: E402

SETTINGS = {"deploy_T0.7_k20_p0.95": dict(temp=0.7, top_k=20, top_p=0.95), "T1": dict(temp=1.0)}
SPEC = ("correct", "resample_target", "accept_untransformed", "residual_untransformed", "bonus_from_draft",
        "argmax_on_reject", "target_kv_stale")
SHIFTS = (0.01, 0.02, 0.05)


def per_chain(z, ch, B):
    o = np.argsort(ch, kind="stable")
    zc, cc = z[o], ch[o]
    b = np.searchsorted(cc, np.arange(B + 1))
    return [zc[b[i]:b[i + 1]] for i in range(B)]


def main(reps=200, out_name="toy_l3.json", B=48, L=256, K=3, n_perm=500, seed=0, sigma=0.3):
    t0 = time.time()
    model = toy.ToyLM.random(V=48, seed=seed)
    rng0 = np.random.default_rng(seed + 5)
    ref = toy.ToyLM(model.target_logits + sigma * rng0.standard_normal(model.target_logits.shape), model.draft_logits)
    R, _ = ref.tables(temp=1.0)
    fmR = toy.pit_tables(R)
    res = {"reps": reps, "B": B, "L": L, "K": K, "n_perm": n_perm, "sigma_ref": sigma, "settings": {}}
    for sname, s in SETTINGS.items():
        P, Q = model.tables(**s)
        P1, Q1 = model.tables(temp=1.0)
        c1v, c2v, _ = toy.simulate_plain(P, 50000, np.random.default_rng(seed + 99))
        ref_tv = float((0.5 * np.abs(R - model.tables(temp=1.0)[0]).sum(-1))[c1v, c2v].mean())
        sres = {"ref_vs_true_T1_mean_tv": ref_tv}
        alts = [("spec", v) for v in SPEC] + [("tempshift", x) for x in SHIFTS]
        for kind, v in alts:
            eff = {}
            if kind == "tempshift":
                Palt, _ = model.tables(**{**s, "temp": s["temp"] * (1 + v)})
                eff["mean_tv"] = float((0.5 * np.abs(Palt - P).sum(-1))[c1v, c2v].mean())
            ps = []
            for r in range(reps):
                rng = np.random.default_rng(seed * 100003 + r * 7 + hash((sname, str(v))) % 1000)
                if kind == "spec":
                    c1, c2, y, ch = toy.simulate_spec(model, P, Q, P1, Q1, K, B * L, v, rng, B=B, return_chain=True)
                else:
                    c1, c2, y, ch = toy.simulate_plain(Palt, B * L, rng, B=B, return_chain=True)
                d1, d2, w, wh = toy.simulate_plain(P, B * L, rng, B=B, return_chain=True)
                z1 = fmR[c1, c2, y] + rng.random(y.size) * R[c1, c2, y]
                z0 = fmR[d1, d2, w] + rng.random(w.size) * R[d1, d2, w]
                vals = per_chain(z1, ch, B) + per_chain(z0, wh, B)
                arm = np.r_[np.ones(B, int), np.zeros(B, int)]
                strata = np.r_[np.arange(B), np.arange(B)]
                t = twosample.permutation_test(strata, arm, vals, n_perm=n_perm, seed=r)
                ps.append(t["p_minp_wy"])
            ps = np.array(ps)
            key = v if kind == "spec" else f"temp_x{1 + v:.2f}"
            sres[key] = eff | {"reject_0.05": float(np.mean(ps <= 0.05)), "reject_0.01": float(np.mean(ps <= 0.01)),
                               "median_p": float(np.median(ps))}
            print(f"[{time.time()-t0:6.0f}s] {sname:24s} {key:24s} reject@0.01 {sres[key]['reject_0.01']:.3f} "
                  f"@0.05 {sres[key]['reject_0.05']:.3f} {eff}", flush=True)
        res["settings"][sname] = sres
    (ROOT / "results" / out_name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, sys.argv[2] if len(sys.argv) > 2 else "toy_l3.json")
