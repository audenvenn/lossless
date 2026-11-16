"""R2: mlx-lm PR #990 (native MTP) at historical commits — generation pass.

Runs inside engines/.venv-mtp with engines/mlx-lm-mtp checked out at the commit under test (see run_r2.sh).
For spec=on uses mtp_generate_step; for spec=off uses generate_step. Records, per emitted token:
  token, from_draft flag, role (main/draft/bonus/corrected), rejections so far,
  PIT stats vs the distribution the engine itself reported for that position (L1; mlx-lm semantics),
and saves the token sequences for a separate fixed-reference scoring pass (r2_score.py).

Exact white-box L1 (v2, 2026-11-16): at every emitted position we also apply the commit's OWN non-speculative
sampler transform (make_sampler's mask chain, same dtype and device, with categorical_sampling stubbed so no RNG is
consumed) to the target logprobs. fm/py/ent/m2 are computed against that engine-own distribution, so tie-breaking and
precision are the engine's, and a support violation means the engine emitted a token its own non-speculative sampler
could never emit. The previous re-implementation (claimed_logits) is kept as fm_claimed/py_claimed. The kept set and
the T=1 top-64 log-probs are stored for an offline transform-conformance check (lossless.declared).

Note: commits before ec584b334f yield the DRAFT's logprobs for accepted draft tokens. run_r2.sh applies a
reporting-only patch so that the target's verify logprobs are yielded instead (sampling is unchanged).
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
mx.set_cache_limit(1 << 30)  # cap the MLX buffer cache at 1 GB (shared 24 GB machine; see LOG 2026-11-16 17:32)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import paths  # noqa: E402
from r1_mlxlm_null import SETTINGS, claimed_logits, token_stats, tv, kl_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=paths.QWEN_BF16)
    ap.add_argument("--label", required=True)
    ap.add_argument("--spec", choices=["on", "off"], required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "r2")
    a = ap.parse_args()

    from mlx_lm import load
    import importlib
    G = importlib.import_module("mlx_lm.generate")  # mlx_lm.__init__ shadows the submodule name
    from mlx_lm.sample_utils import make_sampler
    import mlx_lm.sample_utils as SU

    def engine_transform(lp_row):
        """The commit's own non-speculative sampler applied to lp_row (1, V), stopped at the random draw. Returns the
        logits its categorical draw receives: the masked log-probs scaled by the engine's own arithmetic,
        masked * (1.0 / temp) in the same dtype (bf16 here, i.e. multiplied by bf16(1/T) and rounded to bf16).
        categorical_sampling is mx.compile'd, so its internals cannot be captured; we stub it (no RNG use) and
        replay its one elementwise multiply eagerly, assuming the compiled kernel rounds that product like eager
        MLX. (v2.1: v2 divided by T in float64, which differs from the engine by a mean per-token TV of ~3e-4.)"""
        cap = []
        orig = SU.categorical_sampling
        SU.categorical_sampling = lambda logits, t: (cap.append(logits), mx.zeros((1,), dtype=mx.uint32))[1]
        try:
            decl_sampler(lp_row)
        finally:
            SU.categorical_sampling = orig
        return cap[0] * (1.0 / temp)

    temp, top_p, min_p, top_k, think = SETTINGS[a.setting]
    decl_sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k) if temp > 0 else None
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.label}_{a.setting}_spec{a.spec}"
    t0 = time.time()
    model, tok = load(a.model)
    has_mtp = hasattr(model, "mtp") or getattr(model, "supports_mtp", False)
    print(f"[{tag}] loaded in {time.time()-t0:.1f}s, mtp={has_mtp}, peak {mx.get_peak_memory()/2**30:.2f} GB",
          flush=True)
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text())[: a.n_prompts]
    mx.random.seed(a.seed)
    import random as _random
    _random.seed(a.seed)
    np_rng = np.random.default_rng(a.seed)

    def gen_iter(prompt_ids):
        p = mx.array(prompt_ids)
        if a.spec == "off":
            sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
            for y, lp in G.generate_step(p, model, max_tokens=a.max_tokens, sampler=sampler):
                yield y, lp, False
            return
        params = inspect.signature(G.mtp_generate_step).parameters
        if "sampler" in params:
            sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
            it = G.mtp_generate_step(p, model, max_tokens=a.max_tokens, sampler=sampler)
        else:
            it = G.mtp_generate_step(p, model, max_tokens=a.max_tokens, temp=temp, top_p=top_p, top_k=top_k,
                                     min_p=min_p)
        for item in it:
            yield item[0], item[1], bool(item[2])

    seqs = []
    rec = {k: [] for k in ("seq", "step", "token", "from_draft", "role", "n_rej", "fm", "py", "ent", "m2",
                           "fm_claimed", "py_claimed", "ent_claimed", "m2_claimed",
                           "fm_pre", "py_pre", "tv_pre", "kl_pre", "n_kept", "lse_full")}
    kept_ids, kept_vals, top64_ids, top64_vals = [], [], [], []
    ntok = 0
    tgen = 0.0
    sid = 0
    for pi, ptxt in enumerate(prompts):
        msgs = [{"role": "user", "content": ptxt}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        for s in range(a.samples):
            toks, prev_draft, n_rej, lps = [], False, 0, []
            ts = time.time()
            for t, (y, lp, fd) in enumerate(gen_iter(ids)):
                yi = int(y.item()) if hasattr(y, "item") else int(y)
                if fd:
                    role = 1  # accepted draft
                elif t == 0:
                    role = 0  # first backbone token
                elif prev_draft:
                    role = 2  # bonus after an accepted draft
                else:
                    role = 3  # corrected token after a rejection (spec on) / plain token (spec off)
                    if a.spec == "on":
                        n_rej += 1   # NB: includes the current token's own rejection, so n_rej is NOT a valid
                                     # L1 stratum for role-3 tokens; it is used only for descriptive TV bins
                if temp > 0:
                    x = claimed_logits(lp.reshape(-1), temp, top_p, min_p, top_k, "mlxlm")
                    _, fm, py, ent, m2 = token_stats(x, yi)
                    v = mx.stack([fm, py, ent, m2]); mx.eval(v)
                    vc = np.array(v, dtype=np.float64)
                    masked = engine_transform(lp.reshape(1, -1))
                    m_np = np.array(masked.astype(mx.float32)).reshape(-1)
                    lp_np = np.array(lp.astype(mx.float32)).reshape(-1)
                    kept = np.nonzero(np.isfinite(m_np))[0]
                    kv = m_np[kept].astype(np.float64)        # already scaled by the engine (v2.1)
                    w = np.exp(kv - kv.max()); w /= w.sum()
                    j = int(np.searchsorted(kept, yi))       # kept is ascending by id
                    if j < kept.size and kept[j] == yi:       # order: descending value, ties by ascending id
                        before = (kv > kv[j]) | ((kv == kv[j]) & (kept < yi))
                        fm_e, py_e = float(w[before].sum()), float(w[j])
                    else:
                        fm_e, py_e = 1.0, 0.0                   # outside the engine's own support
                    lw = np.log(np.maximum(w, 1e-300))
                    ent_e = float(-(w * lw).sum()); m2_e = float((w * lw * lw).sum())
                    v = np.array([fm_e, py_e, ent_e, m2_e])
                    rec["n_kept"].append(int(kept.size))
                    rec["lse_full"].append(float(np.logaddexp.reduce(lp_np.astype(np.float64))))
                    if kept.size < m_np.size:
                        kept_ids.append(kept.astype(np.int32)); kept_vals.append(m_np[kept].astype(np.float32))
                    else:
                        kept_ids.append(np.zeros(0, np.int32)); kept_vals.append(np.zeros(0, np.float32))
                    t64 = np.argpartition(-lp_np, 63)[:64]
                    t64 = t64[np.lexsort((t64, -lp_np[t64]))]
                    top64_ids.append(t64.astype(np.int32)); top64_vals.append(lp_np[t64].astype(np.float32))
                else:
                    v = np.full(4, np.nan); vc = np.full(4, np.nan)
                    rec["n_kept"].append(1); rec["lse_full"].append(np.nan)
                for k, val in zip(("fm", "py", "ent", "m2"), v):
                    rec[k].append(val)
                for k, val in zip(("fm_claimed", "py_claimed", "ent_claimed", "m2_claimed"), vc):
                    rec[k].append(val)
                rec["seq"].append(sid); rec["step"].append(t); rec["token"].append(yi)
                rec["from_draft"].append(fd); rec["role"].append(role); rec["n_rej"].append(n_rej)
                toks.append(yi)
                if temp > 0:
                    lps.append(lp.reshape(-1))
                prev_draft = fd
                if yi in tok.eos_token_ids:
                    break
            tgen += time.time() - ts
            # clean teacher-forced (prefill, no cache) pass over the finished sequence: diagnostics
            if temp == 0:
                for k in ("fm_pre", "py_pre", "tv_pre", "kl_pre"):
                    rec[k].extend([np.nan] * len(toks))
                seqs.append({"seq": sid, "prompt": pi, "sample": s, "prompt_ids": list(map(int, ids)), "tokens": toks})
                ntok += len(toks); sid += 1
                continue
            full = mx.array(list(ids) + toks)[None]
            lpp = model(full)[0, len(ids) - 1: len(ids) - 1 + len(toks)].astype(mx.float32)
            lpp = lpp - mx.logsumexp(lpp, axis=-1, keepdims=True)
            for t, (yi, lp) in enumerate(zip(toks, lps)):
                xd = claimed_logits(lp, temp, top_p, min_p, top_k, "mlxlm")
                xp = claimed_logits(lpp[t], temp, top_p, min_p, top_k, "mlxlm")
                pd = mx.softmax(xd, axis=-1)
                pp, fmp, pyp, _, _ = token_stats(xp, yi)
                v = mx.stack([fmp, pyp, tv(pd, pp), kl_raw(lp, lpp[t])]); mx.eval(v)
                v = np.array(v, dtype=np.float64)
                for k, val in zip(("fm_pre", "py_pre", "tv_pre", "kl_pre"), v):
                    rec[k].append(val)
            del lpp, lps
            seqs.append({"seq": sid, "prompt": pi, "sample": s, "prompt_ids": list(map(int, ids)), "tokens": toks})
            ntok += len(toks)
            sid += 1
        if pi % 8 == 7:
            print(f"[{tag}] [{time.time()-t0:7.1f}s] prompt {pi+1}/{len(prompts)} tokens={ntok} "
                  f"{ntok/max(tgen,1e-9):.1f} tok/s", flush=True)
    arr = {k: np.asarray(v) for k, v in rec.items()}
    u = np_rng.random(arr["token"].size)
    arr["z"] = arr["fm"] + u * arr["py"]
    arr["z_pre"] = arr["fm_pre"] + np_rng.random(arr["token"].size) * arr["py_pre"]
    if top64_ids:
        arr["kept_off"] = np.concatenate([[0], np.cumsum([k.size for k in kept_ids])]).astype(np.int64)
        arr["kept_ids"] = np.concatenate(kept_ids) if kept_ids else np.zeros(0, np.int32)
        arr["kept_vals"] = np.concatenate(kept_vals) if kept_vals else np.zeros(0, np.float32)
        arr["top64_ids"] = np.stack(top64_ids); arr["top64_vals"] = np.stack(top64_vals)
    arr["seed"] = np.array(a.seed)
    arr["kept_vals_scaled"] = np.array(1)   # v2.1: kept_vals are the engine's temperature-scaled logits
    np.savez_compressed(a.out / f"{tag}.npz", **arr)
    (a.out / f"{tag}.seqs.json").write_text(json.dumps({"label": a.label, "setting": a.setting, "spec": a.spec,
                                                        "seqs": seqs}))
    meta = {"tag": tag, "n_tokens": ntok, "tok_per_s": ntok / max(tgen, 1e-9),
            "accept_rate": float(arr["from_draft"].sum() / max(1, arr["from_draft"].sum() + (arr["role"] == 3).sum()))
            if a.spec == "on" else None,
            "support_violations": int((arr["py"] <= 0).sum()),
            "support_violations_claimed": int((arr["py_claimed"] <= 0).sum()), "seed": a.seed,
            "elapsed": time.time() - t0}
    (a.out / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
