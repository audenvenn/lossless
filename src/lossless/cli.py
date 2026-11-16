"""lossless: distributional conformance tests for LLM samplers and speculative decoding.

Subcommands
  simulate   Power of the PIT battery per bug class on a toy LM with exact conditionals.
  l1         One-sample randomized-PIT test from per-token claimed distributions
             (an .npz with arrays fm, py, ent, m2  -- see `lossless.pit` -- or a JSONL of OpenAI-style
             top_logprobs with the declared sampler applied by `--temp/--top-p/--top-k/--min-p`).
  l3         Exact two-sample test between two arms (e.g. speculation on vs off) using a fixed scorer:
             per-token reference PIT values in two .npz files (arrays z, seq, prompt).
  identity   Same-seed identity check between two reply files (JSONL with prompt, sample, seed, content,
             reasoning): fraction identical and first-divergence positions.

All tests report p-values. A rejection at the chosen alpha is evidence that the engine does not sample from
the distribution it claims (L1) or that the two arms differ in distribution (L3); the noise floor of the
reference must be calibrated with a negative control before interpreting L1 against an approximate reference.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from . import declared, pit, samplers, twosample


def _l1_from_npz(path: str) -> dict:
    a = dict(np.load(path))
    ok = a["py"] > 0
    rng = np.random.default_rng(0)
    # use the PIT values fixed when the data were generated, if stored; never re-randomise stored results
    z = a["z"] if "z" in a else a["fm"] + rng.random(a["py"].size) * a["py"]
    s = -np.log(np.maximum(a["py"], 1e-300))
    var = np.clip(a["m2"] - a["ent"] ** 2, 0, None)
    rep = pit.uniformity_tests(z[ok], surprisal=(s[ok], a["ent"][ok], var[ok]))
    viol = int((~ok).sum())
    return rep.as_dict() | {"support_violations": viol,
                            "reject_bonf_0.01": bool(viol > 0 or pit.bonferroni_reject(rep, 0.01))}


def _entries(rec: dict):
    """Per-token (emitted value, emitted id or None, [(id or None, value), ...]) from one reply record.
    Supports OpenAI-style logprobs (content[i].logprob, top_logprobs[{token, logprob}]) and llama.cpp n_probs
    (content[i].prob/id, top_probs[{id, prob}] or probs[{id, prob}]). Values are T = 1 log-probabilities."""
    lp = rec.get("logprobs") or {}
    for e in lp.get("content") or []:
        if "top_logprobs" in e:
            tops = [(t.get("id"), float(t["logprob"])) for t in e["top_logprobs"]]   # ids when the server sends them
            yield float(e["logprob"]), e.get("id"), tops
        else:
            tl = e.get("top_probs") or e.get("probs") or []
            if not tl:
                continue
            tops = [(t.get("id"), math.log(t["prob"]) if t["prob"] > 0 else -math.inf) for t in tl]
            yield (math.log(e["prob"]) if e.get("prob", 0) > 0 else -math.inf), e.get("id"), tops


def _l1_from_jsonl(path: str, temp: float, top_p: float, top_k: int, min_p: float, semantics: str,
                   eps: float = 1e-5, post_transform: bool = False) -> dict:
    """L1 from per-token top-K reports (paper, Lemma 3 and the screening rule in lossless.declared).

    The reported values must be the engine's T = 1 log-probabilities (pre-transform), unless post_transform=True,
    in which case they are taken as the distribution the engine sampled from (e.g. llama.cpp post_sampling_probs);
    that variant tests only the final draw, not the declared transform. Positions where the declared distribution
    cannot be computed exactly from the report are skipped (the decision uses only the report, never the emitted
    token), and their count is returned per reason.
    """
    rng = np.random.default_rng(0)
    zs, ss, ents, vars_, viol, screened, n = [], [], [], [], 0, {}, 0
    for line in open(path):
        rec = json.loads(line)
        for y_val, y_id, tops in _entries(rec):
            n += 1
            vals = np.array([v for _, v in tops], dtype=np.float64)
            if post_transform:
                d = declared.declare(vals, 1.0, 0, 1.0, 0.0, "hf", has_tail=True, eps=eps)
            else:
                d = declared.declare(vals, temp, top_k, top_p, min_p, semantics, has_tail=True, eps=eps)
            if d.status not in ("ok",):
                key = d.status + (": " + d.reason if d.reason else "")
                screened[key] = screened.get(key, 0) + 1
                continue
            ids = [i for i, _ in tops]
            if y_id is not None and y_id in ids:
                # categorise by the reported top-list value of the emitted id, not by a separately rounded logprob
                y_val, reported = vals[ids.index(y_id)], True
            else:
                reported = (y_id in ids) if y_id is not None else bool(np.any(vals == y_val))
            z, s_, ent, var, bad = declared.pit(d, y_val, reported, rng.random())
            viol += int(bad)
            zs.append(z); ss.append(s_); ents.append(ent); vars_.append(var)
    if not zs:
        return {"tokens": n, "tested": 0, "screened": screened, "error": "no position could be tested exactly"}
    rep = pit.uniformity_tests(np.array(zs), surprisal=(np.array(ss), np.array(ents), np.array(vars_)))
    return rep.as_dict() | {"tokens": n, "tested": len(zs), "screened": screened, "support_violations": viol,
                            "reject_bonf_0.01": bool(viol > 0 or pit.bonferroni_reject(rep, 0.01)),
                            "post_transform_only": post_transform}


def _load_arm(path: str):
    """Per-sequence reference scores. Formats: (z, off, prompt[, sample]) as written by score_arms.py, or
    per-token (z, seq, prompt) as written by r2_analyze.py."""
    d = np.load(path)
    if "off" in d.files:
        off = d["off"]
        vals = [d["z"][off[i]:off[i + 1]] for i in range(off.size - 1)]
        prompts = d["prompt"].astype(int)
        samples = d["sample"].astype(int) if "sample" in d.files else np.zeros(prompts.size, int)
        return vals, prompts, samples
    vals, prompts = [], []
    for sid in np.unique(d["seq"]):
        m = d["seq"] == sid
        vals.append(d["z"][m]); prompts.append(int(d["prompt"][m][0]))
    return vals, np.array(prompts), np.zeros(len(prompts), int)


def _l3(path1: str, path0: str, n_perm: int) -> dict:
    """Exact permutation test, labels permuted within (prompt, sample) strata; Westfall-Young family p-value."""
    v1, p1, s1 = _load_arm(path1)
    v0, p0, s0 = _load_arm(path0)
    prompt = np.r_[p1, p0]
    arm = np.r_[np.ones(len(v1), int), np.zeros(len(v0), int)]
    return twosample.permutation_test(prompt, arm, v1 + v0, n_perm=n_perm, strata=prompt * 100000 + np.r_[s1, s0])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lossless", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="power per bug class on a toy LM")
    s.add_argument("--reps", type=int, default=40)
    s.add_argument("--out", default="synthetic_power.json")
    a1 = sub.add_parser("l1", help="one-sample randomized-PIT test")
    a1.add_argument("path")
    a1.add_argument("--temp", type=float, default=1.0)
    a1.add_argument("--top-p", type=float, default=1.0)
    a1.add_argument("--top-k", type=int, default=0)
    a1.add_argument("--min-p", type=float, default=0.0)
    a1.add_argument("--semantics", default="hf", choices=["hf", "vllm", "llamacpp", "mlxlm", "mlxlm_legacy"])
    a1.add_argument("--eps", type=float, default=1e-5,
                    help="engine arithmetic tolerance for top-p/min-p boundaries (use ~1e-2 for bf16 engines)")
    a1.add_argument("--post-transform", action="store_true",
                    help="reported probabilities are already post-transform (tests only the final draw)")
    a3 = sub.add_parser("l3", help="exact two-sample test between arms (fixed-scorer PIT values)")
    a3.add_argument("arm1"); a3.add_argument("arm0")
    a3.add_argument("--n-perm", type=int, default=2000)
    idn = sub.add_parser("identity", help="same-seed identity between two reply files")
    idn.add_argument("on"); idn.add_argument("off")
    idn.add_argument("--tokenizer", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "simulate":
        from . import simulate
        return simulate.main(["--reps", str(a.reps), "--out", a.out])
    if a.cmd == "l1":
        res = _l1_from_npz(a.path) if a.path.endswith(".npz") else _l1_from_jsonl(
            a.path, a.temp, a.top_p, a.top_k, a.min_p, a.semantics, eps=a.eps, post_transform=a.post_transform)
    elif a.cmd == "l3":
        res = _l3(a.arm1, a.arm0, a.n_perm)
    elif a.cmd == "identity":
        from . import identity
        return identity.main([a.on, a.off, "--tokenizer", a.tokenizer])
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
