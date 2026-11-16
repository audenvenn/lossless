"""Analyze R2 (mlx-lm PR #990 historical commits): L1 PIT, support violations, contamination dose-response,
and exact two-sample tests against the speculation-off arm using ONE fixed external reference.

Step 1 (GPU, under lock): score every saved sequence with a fixed reference (mlx-lm main, our .venv) ->
         results/r2/<tag>.ref.npz with per-token z_ref (PIT at the declared temperature, no truncation).
Step 2 (CPU): summary table -> results/r2/summary.json
Usage: python -I r2_analyze.py score   (under the GPU lock)    |    python -I r2_analyze.py report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import paths  # noqa: E402
from lossless import pit, twosample  # noqa: E402

R2 = ROOT / "results" / "r2"
REF = paths.QWEN_BF16   # bf16 HF checkpoint, loaded by mlx-lm main


def score():
    import mlx.core as mx
    mx.set_cache_limit(1 << 30)  # cap the MLX buffer cache (shared machine)
    from mlx_lm import load
    from r1_mlxlm_null import SETTINGS
    model, tok = load(REF)
    rng = np.random.default_rng(0)
    for f in sorted(R2.glob("*.seqs.json")):
        out = f.with_name(f.name.replace(".seqs.json", ".ref.npz"))
        if out.exists():
            continue
        d = json.loads(f.read_text())
        temp = SETTINGS[d["setting"]][0]
        zs, seq_ids, lens = [], [], []
        for s in d["seqs"]:
            ids, toks = s["prompt_ids"], s["tokens"]
            if not toks:
                continue
            logits = model(mx.array(ids + toks)[None])[0, len(ids) - 1: len(ids) - 1 + len(toks)]
            p = mx.softmax(logits.astype(mx.float32) / temp, axis=-1)
            y = mx.array(toks)
            py = mx.take_along_axis(p, y[:, None], axis=-1)
            fm = mx.sum(mx.where(p > py, p, 0.0), axis=-1)
            mx.eval(py, fm)
            zs.append(np.array(fm) + rng.random(len(toks)) * np.array(py)[:, 0])
            seq_ids.append(s["seq"]); lens.append(len(toks))
        np.savez_compressed(out, z=np.concatenate(zs), seq=np.repeat(seq_ids, lens),
                            prompt=np.repeat([s["prompt"] for s in d["seqs"] if s["tokens"]], lens))
        print("scored", out.name, sum(lens), "tokens", flush=True)


def l1_report(arr):
    s = -np.log(np.maximum(arr["py"], 1e-300))
    var = np.clip(arr["m2"] - arr["ent"] ** 2, 0, None)
    ok = arr["py"] > 0
    out = {}
    strata = {"all": np.ones_like(ok), "verified(draft-accepted+corrected)": np.isin(arr["role"], [1, 3]),
              "bonus": arr["role"] == 2}
    for name, m in strata.items():
        m = m & ok
        if m.sum() < 50:
            continue
        rep = pit.uniformity_tests(arr["z"][m], surprisal=(s[m], arr["ent"][m], var[m]))
        out[name] = {"n": int(m.sum()), "reject_bonf_0.01": bool(pit.bonferroni_reject(rep, 0.01)),
                     "ks_p": rep.ks_p, "mean_z": rep.mean_z, "mean_z_p": rep.mean_z_p,
                     "surprisal_z": rep.surprisal_z, "min_p": rep.min_p}
    out["support_violations"] = int((~ok).sum())
    return out


def contamination(arr):
    tvp = arr["tv_pre"]
    bins = [(0, 0), (1, 5), (6, 20), (21, 10**9)]
    res = {"mean_tv_vs_prefill": float(tvp.mean()), "p99_tv_vs_prefill": float(np.quantile(tvp, 0.99))}
    for lo, hi in bins:
        m = (arr["n_rej"] >= lo) & (arr["n_rej"] <= hi)
        if m.sum():
            res[f"tv_nrej_{lo}-{hi if hi < 10**9 else 'inf'}"] = {"n": int(m.sum()), "mean": float(tvp[m].mean())}
    return res


def report():
    runs = {}
    for f in sorted(R2.glob("*.npz")):
        if f.name.endswith(".ref.npz"):
            continue
        tag = f.name[:-4]
        arr = dict(np.load(f))
        runs[tag] = {"l1": l1_report(arr), "contamination": contamination(arr)}
        meta = R2 / f"{tag}.meta.json"
        if meta.exists():
            runs[tag]["meta"] = json.loads(meta.read_text())
    # two-sample vs D_head spec-off, same setting, fixed reference
    for tag in list(runs):
        if not tag.endswith("_specon"):
            continue
        setting = tag.split("_", 2)[2].rsplit("_spec", 1)[0] if tag.count("_") >= 3 else None
        label = tag.split("_")[0] + "_" + tag.split("_")[1]
        setting = tag[len(label) + 1: -len("_specon")]
        off = R2 / f"D_head_{setting}_specoff.ref.npz"
        on = R2 / f"{tag}.ref.npz"
        if not (off.exists() and on.exists()):
            continue
        A, B = np.load(on), np.load(off)
        vals, prompts, arms = [], [], []
        for arm, d in ((1, A), (0, B)):
            for sid in np.unique(d["seq"]):
                m = d["seq"] == sid
                vals.append(d["z"][m]); prompts.append(int(d["prompt"][m][0])); arms.append(arm)
        runs[tag]["two_sample_vs_off"] = twosample.permutation_test(np.array(prompts), np.array(arms), vals,
                                                                     n_perm=2000)
    (R2 / "summary.json").write_text(json.dumps(runs, indent=1, default=float))
    for tag, r in runs.items():
        l1 = r["l1"].get("all", {})
        ts = r.get("two_sample_vs_off", {}).get("p", {})
        print(f"{tag:55s} L1 reject={l1.get('reject_bonf_0.01')} n={l1.get('n')} meanz={l1.get('mean_z', float('nan')):.4f} "
              f"supviol={r['l1']['support_violations']} | TVpre={r['contamination']['mean_tv_vs_prefill']:.4f} | "
              f"2samp p={ {k: round(v, 4) for k, v in ts.items()} }")


if __name__ == "__main__":
    {"score": score, "report": report}[sys.argv[1]]()
