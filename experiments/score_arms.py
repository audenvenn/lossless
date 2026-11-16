"""Score arm files ONCE with a fixed reference model and cache per-token values for CPU-side L3 comparisons.

Inputs: L3 reply files (results/l3/*.jsonl: re-tokenised completion text) or R2 sequence files (*.seqs.json: exact
token ids). For every sequence: per-token randomised PIT under the reference at the setting's temperature without
truncation (the same scoring function as l3_score.py / r2_analyze.py; any fixed function keeps L3 exact).
Output: <out_dir>/<source dir>__<file stem>.npz with z and rank (all tokens), off (sequence offsets), prompt, sample,
length.

Usage (under the GPU lock): python -I score_arms.py --ref PATH --out-dir results/zref/mlx4 FILE [FILE ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
mx.set_cache_limit(1 << 30)  # shared 24 GB machine: cap the MLX buffer cache (see LOG 2026-11-16 17:32)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from l3_client import SETTINGS  # noqa: E402
from l3_score import completion_text  # noqa: E402


def setting_of(path: Path, meta: dict | None) -> str:
    if meta and meta.get("setting"):
        return meta["setting"]
    for s in SETTINGS:
        if s in path.name:
            return s
    raise ValueError(f"cannot infer setting from {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-score-tokens", type=int, default=256)
    a = ap.parse_args()
    from mlx_lm import load
    model, tok = load(a.ref)
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for f in a.files:
        # prefix the source directory: results/r2 and results/r2x contain files with identical names
        out = a.out_dir / (f"{f.parent.name}__" + f.name.replace(".seqs.json", "").replace(".jsonl", "") + ".npz")
        if out.exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        rng = np.random.default_rng(0)
        items = []                                    # (prompt, sample, prompt_ids, completion_ids)
        if f.name.endswith(".seqs.json"):
            d = json.loads(f.read_text())
            S = setting_of(f, d)
            for s in d["seqs"]:
                items.append((s["prompt"], s.get("sample", 0), s["prompt_ids"], s["tokens"][: a.max_score_tokens]))
        else:
            S = setting_of(f, None)
            think = SETTINGS[S][4]
            for line in f.read_text().splitlines():
                rec = json.loads(line)
                msgs = [{"role": "user", "content": prompts[rec["prompt"]]}]
                try:
                    pid = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
                except TypeError:
                    pid = tok.apply_chat_template(msgs, add_generation_prompt=True)
                ctoks = tok.encode(completion_text(rec), add_special_tokens=False)[: a.max_score_tokens]
                items.append((rec["prompt"], rec.get("sample", 0), pid, ctoks))
        temp = SETTINGS[S][0] or 1.0
        zs, rks, pr, sa, ln = [], [], [], [], []
        for p, smp, pid, ctoks in items:
            if not ctoks:
                continue
            logits = model(mx.array(list(pid) + list(ctoks))[None])[0, len(pid) - 1: len(pid) - 1 + len(ctoks)]
            prob = mx.softmax(logits.astype(mx.float32) / temp, axis=-1)
            y = mx.array(list(ctoks))
            py = mx.take_along_axis(prob, y[:, None], axis=-1)
            fm = mx.sum(mx.where(prob > py, prob, 0.0), axis=-1)
            rk = mx.sum(prob > py, axis=-1)                      # tokens strictly more likely than the emitted one
            mx.eval(py, fm, rk)
            zs.append(np.array(fm) + rng.random(len(ctoks)) * np.array(py)[:, 0])
            rks.append(np.array(rk).astype(np.int32))
            pr.append(p); sa.append(smp); ln.append(len(ctoks))
            mx.clear_cache()
        np.savez_compressed(out, z=np.concatenate(zs), rank=np.concatenate(rks), off=np.r_[0, np.cumsum(ln)], prompt=np.array(pr),
                            sample=np.array(sa), length=np.array(ln), setting=S, ref=str(Path(a.ref).name))
        print(f"wrote {out.name}: {len(ln)} seqs, {sum(ln)} tokens, setting {S}", flush=True)


if __name__ == "__main__":
    main()
