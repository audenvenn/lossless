"""R2x analysis (CPU): exact white-box L1, transform conformance, and prefill consistency per run.

L1 (exact): emitted tokens against the commit's OWN non-speculative sampler transform of the target log-probs at
that position (captured in r2_mlx_mtp.py). Strata are fixed before the token is drawn: 'draft' positions (a draft
was proposed: accepted or corrected, roles 1 and 3) and 'bonus' positions (after an accepted draft, role 2). A
support violation is a token the engine's own non-speculative sampler could never emit, so any violation rejects.

Transform conformance (deterministic, no statistics): the engine's transform vs the exact declared semantics
(lossless.declared, float64, top-64 report + full-vocabulary normaliser). Reported as per-position class-level TV.

Usage: python -I r2x_analyze.py [results/r2x]  -> <dir>/summary_l1.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from lossless import declared, pit  # noqa: E402
from l3_client import SETTINGS  # noqa: E402


def bf16(x):
    """Round float32 values to bfloat16 (round to nearest even), returned as float32."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


def engine_actual(d, temp):
    """v2 files stored the masked log-probs and divided by T in float64; the engine actually draws from
    softmax(bf16(masked * bf16(1/T))). Recompute F(y-), p(y) (and entropy terms) under that distribution, re-using
    the generation-time uniforms recovered as u = (z - fm) / py, so nothing is re-randomised. Returns None when
    the file already stores the engine's scaled logits (v2.1) or no truncation was active."""
    if "kept_vals_scaled" in d.files or "kept_off" not in d.files or d["kept_ids"].size == 0:
        return None
    off, kid, kval, tok = d["kept_off"], d["kept_ids"], d["kept_vals"], d["token"]
    sc = bf16(np.float32(1.0 / temp))
    n = tok.size
    fm2, py2, ent2, m22 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        ki, kv = kid[off[i]:off[i + 1]], kval[off[i]:off[i + 1]]
        if ki.size == 0:
            fm2[i], py2[i] = d["fm"][i], d["py"][i]; ent2[i], m22[i] = d["ent"][i], d["m2"][i]
            continue
        a = bf16(bf16(kv) * sc).astype(np.float64)
        w = np.exp(a - a.max()); w /= w.sum()
        lw = np.log(np.maximum(w, 1e-300))
        ent2[i] = -(w * lw).sum(); m22[i] = (w * lw * lw).sum()
        j = np.nonzero(ki == tok[i])[0]
        if j.size:
            j = int(j[0])
            before = (a > a[j]) | ((a == a[j]) & (ki < tok[i]))
            fm2[i], py2[i] = w[before].sum(), w[j]
        else:
            fm2[i], py2[i] = 1.0, 0.0
    ok = d["py"] > 0
    u = np.where(ok, (d["z"] - d["fm"]) / np.where(ok, d["py"], 1.0), 0.0)
    u = np.clip(u, 0.0, 1.0)
    return {"fm": fm2, "py": py2, "ent": ent2, "m2": m22, "z": fm2 + u * py2}


def derandomized(fm, py, ok, B=200, seed=4242):
    """Vovk-Wang merging: twice the arithmetic mean of p-values over B pre-specified independent uniform draws is
    a valid p-value; removes the dependence of the KS/CvM/chi2/mean-z decisions on one draw of the tester's coin."""
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(B):
        z = (fm + rng.random(fm.size) * py)[ok]
        r = pit.uniformity_tests(z).as_dict()
        for k in ("ks_p", "cvm_p", "chi2_p", "mean_z_p"):
            if k in r and r[k] == r[k]:
                acc.setdefault(k, []).append(r[k])
    return {k: float(min(1.0, 2 * np.mean(v))) for k, v in acc.items()}


def battery(z, s=None):
    rep = pit.uniformity_tests(z, surprisal=s)
    return rep.as_dict() | {"n": int(z.size), "reject_bonf_0.01": bool(pit.bonferroni_reject(rep, 0.01))}


def l1(d, temp=None):
    ea = engine_actual(d, temp) if (temp and hasattr(d, "files")) else None
    if "py_claimed" not in d.files:      # R2 v1 files: one set of statistics (exact at T = 1, no truncation)
        d = dict(d) | {"fm_claimed": d["fm"], "py_claimed": d["py"]}
    if ea is not None:                   # v2 files: switch to the engine's actual bf16 arithmetic
        d = dict(d) | ea
    keys = set(d.files) if hasattr(d, "files") else set(d)
    rng = np.random.default_rng(12345)
    # primary analysis: the PIT values fixed at generation time (uniforms drawn from the run's documented seed),
    # never re-randomised after looking at the data
    z = d["z"] if "z" in keys else d["fm"] + rng.random(d["fm"].size) * d["py"]
    zc = d["fm_claimed"] + rng.random(d["fm"].size) * d["py_claimed"]
    sur = (-np.log(np.maximum(d["py"], 1e-300)), -d["ent"] * -1 if False else d["ent"], np.maximum(d["m2"] - d["ent"] ** 2, 0))
    role = d["role"]
    # impossible tokens (py = 0) are counted separately and reject on their own; under H0 there are none, so leaving
    # them out of the battery does not change its null distribution (and keeps -log 0 out of the surprisal sum)
    ok = d["py"] > 0
    strata = {"all": ok, "draft(1,3)": np.isin(role, (1, 3)) & ok, "bonus(2)": (role == 2) & ok}
    out = {}
    for k, m in strata.items():
        if m.sum() < 50:
            continue
        out[k] = battery(z[m], tuple(x[m] for x in sur))
    out["support_violations"] = int((d["py"] <= 0).sum())
    out["engine_arithmetic"] = "bf16-emulated (v2 file)" if ea is not None else "as captured"
    out["derandomized_all"] = derandomized(d["fm"], d["py"], ok)
    # entropy screen (fixed before the draw): positions whose largest declared probability is <= 0.9
    lowconf = d["ent"] > 0.325          # H(0.9, 0.1) = 0.325 nats: excludes near-deterministic positions
    m = ok & lowconf
    if m.sum() >= 50:
        out["entropy_screened"] = battery(z[m], tuple(x[m] for x in sur)) | {"positions": int(m.sum())}
    # exact fixed-budget version (Lemma 2): the first M = n_prompts * max_tokens values, padded with fresh uniforms
    M = 48 * 256
    zz = z[ok][:M]
    if zz.size < M:
        zz = np.r_[zz, np.random.default_rng(777).random(M - zz.size)]
    rep = pit.uniformity_tests(zz)
    out["all_fixed_budget"] = {"M": M, "padded": int(M - min(M, int(ok.sum()))), "ks_p": rep.as_dict()["ks_p"],
                               "reject_bonf_0.01": bool(pit.bonferroni_reject(rep, 0.01))}
    okc = (d["py_claimed"] > 0) & np.isfinite(zc)
    out["legacy_reimplementation"] = {"support_violations": int((d["py_claimed"] <= 0).sum()), "all": battery(zc[okc])}
    out["violations_by_role"] = {int(r): int(((d["py"] <= 0) & (role == r)).sum()) for r in np.unique(role)}
    return out


def transform_conformance(d, setting):
    temp, top_p, min_p, top_k, _ = SETTINGS[setting]
    if "top64_vals" not in d.files or not (0 < top_p < 1 or top_k > 0 or min_p > 0):
        return None
    off, kid, kval = d["kept_off"], d["kept_ids"], d["kept_vals"]
    scaled = "kept_vals_scaled" in d.files and int(d["kept_vals_scaled"]) == 1
    tvs, n_amb, n_ins, n_diff = [], 0, 0, 0
    for i in range(d["top64_vals"].shape[0]):
        v = d["top64_vals"][i].astype(np.float64) - d["lse_full"][i]
        ids = d["top64_ids"][i]
        dec = declared.declare(v, temp, top_k, top_p, min_p, "mlxlm_legacy", has_tail=True, eps=1e-9)
        if dec.status == "insufficient":
            n_ins += 1
            continue
        n_amb += dec.status == "ambiguous"
        ki, kv = kid[off[i]:off[i + 1]], kval[off[i]:off[i + 1]].astype(np.float64)
        sc = kv if scaled else kv / temp                      # v2.1 files store the engine's scaled logits
        w = np.exp(sc - sc.max()); w /= w.sum()
        val_of = dict(zip(ids.tolist(), v.tolist()))
        eng = np.zeros(dec.mass.size)
        for t, wt in zip(ki.tolist(), w):
            c = dec.category(val_of.get(t), t in val_of)
            if c < 0:
                c = None
            if c is None:
                eng = np.append(eng, wt) if eng.size == dec.mass.size else eng + np.r_[np.zeros(dec.mass.size), wt]
            else:
                eng[c] += wt
        ref = np.r_[dec.mass, np.zeros(eng.size - dec.mass.size)]
        tv = 0.5 * float(np.abs(eng - ref).sum())
        tvs.append(tv)
        n_diff += tv > 1e-6
    tvs = np.array(tvs)
    return {"positions": int(d["top64_vals"].shape[0]), "screened_insufficient": n_ins, "ambiguous": int(n_amb),
            "frac_positions_differing": float(n_diff / max(1, tvs.size)), "mean_tv": float(tvs.mean()),
            "p99_tv": float(np.quantile(tvs, 0.99)), "max_tv": float(tvs.max())}


def main(rdir="results/r2x"):
    R = ROOT / rdir
    out = {}
    for f in sorted(R.glob("*.npz")):
        if f.name.endswith(".ref.npz"):
            continue
        d = np.load(f)
        files = d.files
        meta = json.loads((R / f.name.replace(".npz", ".meta.json")).read_text())
        setting = next(s for s in SETTINGS if s in f.name)
        res = {"seed": int(d["seed"]) if "seed" in d.files else None, "tokens": int(d["token"].size),
               "accept_rate": meta.get("accept_rate")}
        if SETTINGS[setting][0] > 0:
            res["l1"] = l1(d, SETTINGS[setting][0])
            res["transform_conformance"] = transform_conformance(d, setting)
            tvp = d["tv_pre"]
            res["prefill_tv"] = {"mean": float(np.nanmean(tvp)),
                                 "by_rejections": {f"{a}-{b}": float(np.nanmean(tvp[(d['n_rej'] >= a) & (d['n_rej'] <= b)]))
                                                   for a, b in ((0, 0), (1, 5), (6, 20), (21, 10**9))
                                                   if ((d['n_rej'] >= a) & (d['n_rej'] <= b)).any()}}
        out[f.stem] = res
        l1a = res.get("l1", {}).get("all", {})
        print(f"{f.stem:46s} n={res['tokens']:6d} L1 all: KS p {l1a.get('ks_p', float('nan')):.3g} "
              f"surpr z {l1a.get('surprisal_z', float('nan')):+.2f} viol {res.get('l1', {}).get('support_violations')} "
              f"(legacy viol {res.get('l1', {}).get('legacy_reimplementation', {}).get('support_violations')}) "
              f"reject {l1a.get('reject_bonf_0.01')}  prefill TV {res.get('prefill_tv', {}).get('mean', float('nan')):.4f}",
              flush=True)
    (R / "summary_l1.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/r2x")
