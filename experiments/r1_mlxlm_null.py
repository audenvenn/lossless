"""R1: mlx-lm non-speculative baseline on a real model (PIT null, noise floor, sampler-semantics gap).

For each prompt and sampler setting, generate with mlx-lm's own sampler (make_sampler) and capture the exact
log-probabilities the sampler received at every step. Then, per emitted token:
  z_dec : PIT against the claimed distribution computed by an INDEPENDENT implementation of mlx-lm's documented
          semantics (truncate on T=1 probs: top_p -> min_p -> top_k, then temperature) from the decode-time
          logprobs. Under a correct sampler this is exactly Unif(0,1).
  z_hf  : PIT of the same tokens against HF/vLLM semantics (temperature first) from the same logprobs.
  z_pre : PIT against the claimed distribution computed from a teacher-forced (prefill) pass over the finished
          sequence -- the noise floor a gray-box reference would face.
plus per-token TV(claimed_dec, claimed_hf), TV(claimed_dec, claimed_pre), KL(raw_dec || raw_pre).

Run (optionally under a GPU lock wrapper):  python -I r1_mlxlm_null.py ...
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import paths  # noqa: E402
import sys
import time
from pathlib import Path

import mlx.core as mx
mx.set_cache_limit(1 << 30)  # cap the MLX buffer cache at 1 GB (shared 24 GB machine; see LOG 2026-11-16 17:32)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lossless import pit  # noqa: E402

SETTINGS = {
    # name: (temp, top_p, min_p, top_k, enable_thinking)
    "T1.0_plain": (1.0, 1.0, 0.0, 0, False),
    "qwen_nothink_T0.7_p0.8_k20": (0.7, 0.8, 0.0, 20, False),
    "qwen_think_T0.6_p0.95_k20": (0.6, 0.95, 0.0, 20, True),
    "greedy": (0.0, 1.0, 0.0, 0, False),
}


# ---------- independent implementations of the claimed distributions (float32, on GPU) ----------
def _mask_top_p(z: mx.array, p: float) -> mx.array:
    """Keep the smallest top set with mass >= p (crossing token kept). z: log-weights (may contain -inf)."""
    if not (0.0 < p < 1.0):
        return z
    probs = mx.softmax(z, axis=-1)
    order = mx.argsort(-z, axis=-1)
    sp = mx.take_along_axis(probs, order, axis=-1)
    cum = mx.cumsum(sp, axis=-1)
    n_keep = mx.sum(cum < p, axis=-1, keepdims=True) + 1
    thr = mx.take_along_axis(mx.take_along_axis(z, order, axis=-1), n_keep - 1, axis=-1)
    return mx.where(z < thr, -mx.inf, z)


def _mask_top_k(z: mx.array, k: int) -> mx.array:
    if k <= 0 or k >= z.shape[-1]:
        return z
    kth = mx.sort(z, axis=-1)[..., -k][..., None]
    return mx.where(z < kth, -mx.inf, z)


def _mask_min_p(z: mx.array, m: float) -> mx.array:
    if m <= 0:
        return z
    lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
    thr = mx.max(lp, axis=-1, keepdims=True) + np.log(m)
    return mx.where(lp < thr, -mx.inf, z)


def claimed_logits(logprobs: mx.array, temp, top_p, min_p, top_k, semantics: str) -> mx.array:
    z = logprobs.astype(mx.float32)
    if semantics == "mlxlm":       # truncation on T=1 log-probs, temperature last
        z = _mask_top_p(z, top_p)
        z = _mask_min_p(z, min_p)
        z = _mask_top_k(z, top_k)
        return z / temp
    if semantics == "hf":          # temperature first (transformers, vLLM ordering of top_k/top_p)
        z = z / temp
        z = _mask_top_k(z, top_k)
        z = _mask_top_p(z, top_p)
        z = _mask_min_p(z, min_p)
        return z
    raise ValueError(semantics)


def token_stats(x: mx.array, y: int):
    """From claimed logits x (V,): F(y-), p(y), entropy, var of surprisal, and the prob vector."""
    p = mx.softmax(x, axis=-1)
    py = p[y]
    ids = mx.arange(p.shape[-1])
    fminus = mx.sum(mx.where((p > py) | ((p == py) & (ids < y)), p, 0.0))
    logp = mx.where(p > 0, mx.log(mx.maximum(p, 1e-38)), 0.0)
    ent = -mx.sum(p * logp)
    m2 = mx.sum(p * logp * logp)
    return p, fminus, py, ent, m2


def tv(p, q):
    return 0.5 * mx.sum(mx.abs(p - q))


def kl_raw(lp_a: mx.array, lp_b: mx.array):
    a = lp_a.astype(mx.float32); b = lp_b.astype(mx.float32)
    a = a - mx.logsumexp(a, axis=-1, keepdims=True)   # renormalize: decode logprobs may be bf16-rounded
    b = b - mx.logsumexp(b, axis=-1, keepdims=True)
    pa = mx.exp(a)
    return mx.sum(mx.where(pa > 0, pa * (a - b), 0.0))


def run(model_path: str, out_dir: Path, n_prompts: int, max_tokens: int, settings: list[str], seed: int):
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text())[:n_prompts]
    t0 = time.time()
    model, tok = load(model_path)
    print(f"loaded {model_path} in {time.time()-t0:.1f}s; peak mem {mx.get_peak_memory()/2**30:.2f} GB", flush=True)
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    summary = {"model": model_path, "max_tokens": max_tokens, "settings": {}}
    for sname in settings:
        temp, top_p, min_p, top_k, think = SETTINGS[sname]
        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
        rec = {k: [] for k in ("prompt", "step", "token", "fm_dec", "py_dec", "ent_dec", "m2_dec",
                               "fm_hf", "py_hf", "ent_hf", "m2_hf", "fm_pre", "py_pre", "ent_pre", "m2_pre",
                               "tv_hf", "tv_pre", "kl_pre")}
        ntok = 0
        tgen = 0.0
        for pi, ptxt in enumerate(prompts):
            msgs = [{"role": "user", "content": ptxt}]
            try:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
            except TypeError:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
            prompt = mx.array(ids)
            toks, lps = [], []
            ts = time.time()
            for (y, lp) in generate_step(prompt, model, max_tokens=max_tokens, sampler=sampler):
                yi = int(y.item()) if hasattr(y, "item") else int(y)
                toks.append(yi)
                lps.append(lp.reshape(-1))
                if yi in tok.eos_token_ids:
                    break
            tgen += time.time() - ts
            # teacher-forced (prefill) pass over prompt + generated tokens
            full = mx.array(ids + toks)[None]
            logits = model(full)
            lp_pre_all = logits[0, len(ids) - 1: len(ids) - 1 + len(toks)].astype(mx.float32)
            lp_pre_all = lp_pre_all - mx.logsumexp(lp_pre_all, axis=-1, keepdims=True)
            for t, (yi, lp) in enumerate(zip(toks, lps)):
                xd = claimed_logits(lp, temp, top_p, min_p, top_k, "mlxlm")
                xh = claimed_logits(lp, temp, top_p, min_p, top_k, "hf")
                xp = claimed_logits(lp_pre_all[t], temp, top_p, min_p, top_k, "mlxlm")
                pd, fd, yd, ed, md = token_stats(xd, yi)
                ph, fh, yh, eh, mh = token_stats(xh, yi)
                pp, fp, yp, ep, mp = token_stats(xp, yi)
                vals = mx.stack([fd, yd, ed, md, fh, yh, eh, mh, fp, yp, ep, mp,
                                 tv(pd, ph), tv(pd, pp), kl_raw(lp, lp_pre_all[t])])
                mx.eval(vals)
                v = np.array(vals, dtype=np.float64)
                for k, val in zip(("fm_dec", "py_dec", "ent_dec", "m2_dec", "fm_hf", "py_hf", "ent_hf", "m2_hf",
                                   "fm_pre", "py_pre", "ent_pre", "m2_pre", "tv_hf", "tv_pre", "kl_pre"), v):
                    rec[k].append(val)
                rec["prompt"].append(pi); rec["step"].append(t); rec["token"].append(yi)
            ntok += len(toks)
            del logits, lp_pre_all, lps
            if pi % 8 == 7:
                print(f"[{time.time()-t0:7.1f}s] {sname} prompt {pi+1}/{len(prompts)} tokens={ntok} "
                      f"gen {ntok/max(tgen,1e-9):.1f} tok/s peak {mx.get_peak_memory()/2**30:.2f} GB", flush=True)
        arr = {k: np.asarray(v) for k, v in rec.items()}
        u = rng.random((3, arr["token"].size))
        res = {}
        for tag, j in (("dec", 0), ("hf", 1), ("pre", 2)):
            z = arr[f"fm_{tag}"] + u[j] * arr[f"py_{tag}"]
            arr[f"z_{tag}"] = z
            s = -np.log(np.maximum(arr[f"py_{tag}"], 1e-300))
            var = np.clip(arr[f"m2_{tag}"] - arr[f"ent_{tag}"] ** 2, 0, None)
            rep = pit.uniformity_tests(z, surprisal=(s, arr[f"ent_{tag}"], var))
            res[tag] = rep.as_dict() | {"reject_bonf_0.01": bool(pit.bonferroni_reject(rep, 0.01))}
        for tag in ("dec", "hf", "pre"):
            res[f"support_violations_{tag}"] = int((arr[f"py_{tag}"] <= 0).sum())
        res["mean_tv_hf_vs_mlxlm"] = float(arr["tv_hf"].mean())
        res["mean_tv_prefill_vs_decode"] = float(arr["tv_pre"].mean())
        res["p99_tv_prefill_vs_decode"] = float(np.quantile(arr["tv_pre"], 0.99))
        res["mean_kl_raw_decode_prefill"] = float(arr["kl_pre"].mean())
        res["n_tokens"] = int(arr["token"].size)
        res["gen_tok_per_s"] = float(ntok / max(tgen, 1e-9))
        summary["settings"][sname] = res
        np.savez_compressed(out_dir / f"r1_{sname}.npz", **arr)
        print(json.dumps({sname: res}, indent=1), flush=True)
    (out_dir / "r1_summary.json").write_text(json.dumps(summary, indent=1))
    print("done", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=paths.QWEN_MLX4)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "r1")
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    run(a.model, a.out, a.n_prompts, a.max_tokens, a.settings.split(","), a.seed)
