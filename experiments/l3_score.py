"""Score L3 replies of two arms with ONE fixed reference model and run the exact stratified permutation test.

The reference only defines a scoring function (per-token randomized PIT under the reference at the declared
temperature, no truncation); it does not need to match the engine's weights or numerics for the test to be
exact. Closer references give more power.

Usage (under the GPU lock): python -I l3_score.py --arm-files A_on.jsonl A_off.jsonl --setting S --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
mx.set_cache_limit(1 << 30)  # cap the MLX buffer cache at 1 GB (shared 24 GB machine; see LOG 2026-11-16 17:32)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import paths  # noqa: E402
from lossless import twosample  # noqa: E402
from l3_client import SETTINGS  # noqa: E402


def completion_text(rec: dict) -> str:
    c, r = rec.get("content") or "", rec.get("reasoning") or ""
    if r:
        return r.strip() + "\n</think>\n\n" + c.lstrip()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-files", nargs=2, type=Path, required=True, help="arm1 (e.g. spec on) then arm0 (off)")
    ap.add_argument("--setting", required=True)
    ap.add_argument("--ref", default=paths.QWEN_MLX4)
    ap.add_argument("--max-score-tokens", type=int, default=256)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from mlx_lm import load
    temp, top_p, min_p, top_k, think = SETTINGS[a.setting]
    model, tok = load(a.ref)
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    rng = np.random.default_rng(0)
    prompt_ids, arms, zs, lens, finish = [], [], [], [], []
    for arm_val, path in ((1, a.arm_files[0]), (0, a.arm_files[1])):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            msgs = [{"role": "user", "content": prompts[rec["prompt"]]}]
            try:
                pid = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
            except TypeError:
                pid = tok.apply_chat_template(msgs, add_generation_prompt=True)
            ctoks = tok.encode(completion_text(rec), add_special_tokens=False)[: a.max_score_tokens]
            if not ctoks:
                continue
            logits = model(mx.array(list(pid) + ctoks)[None])[0, len(pid) - 1: len(pid) - 1 + len(ctoks)]
            x = logits.astype(mx.float32) / temp
            p = mx.softmax(x, axis=-1)
            y = mx.array(ctoks)
            py = mx.take_along_axis(p, y[:, None], axis=-1)
            fm = mx.sum(mx.where(p > py, p, 0.0), axis=-1)
            mx.eval(py, fm)
            z = np.array(fm) + rng.random(len(ctoks)) * np.array(py)[:, 0]
            prompt_ids.append(rec["prompt"]); arms.append(arm_val); zs.append(z); lens.append(len(ctoks))
            finish.append(rec.get("finish_reason"))
    res = twosample.permutation_test(np.array(prompt_ids), np.array(arms), zs, n_perm=a.n_perm)
    res |= {"setting": a.setting, "ref": a.ref, "files": [str(f) for f in a.arm_files],
            "mean_len_arm1": float(np.mean([l for l, r in zip(lens, arms) if r == 1])),
            "mean_len_arm0": float(np.mean([l for l, r in zip(lens, arms) if r == 0]))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
