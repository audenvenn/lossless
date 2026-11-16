"""Black-box support test: the reference-model rank of every emitted token in engine replies.

A sampler that honours "T=1, no truncation" emits tokens beyond rank R at a rate equal to the mean tail mass
beyond R (F7/F8: ~1.1% beyond 32, ~0.1% beyond 1024 for Qwen3.5-4B on these prompts). An engine that silently
truncates to its top-R never does (up to rank disagreements between its weights and the reference, which we
estimate from engines without truncation).

Usage (GPU lock): python -I rank_scan.py out.json file1.jsonl [file2.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
mx.set_cache_limit(1 << 30)  # cap the MLX buffer cache at 1 GB (shared 24 GB machine; see LOG 2026-11-16 17:32)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import paths  # noqa: E402
from l3_score import completion_text  # noqa: E402
from l3_client import SETTINGS  # noqa: E402

THRESH = (20, 32, 64, 1024)


def main():
    out_path, files = Path(sys.argv[1]), sys.argv[2:]
    from mlx_lm import load
    model, tok = load(paths.QWEN_MLX4)
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    res = {}
    for f in files:
        setting = next(s for s in SETTINGS if s in Path(f).name)
        think = SETTINGS[setting][4]
        ranks = []
        for line in Path(f).read_text().splitlines():
            rec = json.loads(line)
            msgs = [{"role": "user", "content": prompts[rec["prompt"]]}]
            pid = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
            ct = tok.encode(completion_text(rec), add_special_tokens=False)[:256]
            if not ct:
                continue
            lg = model(mx.array(list(pid) + ct)[None])[0, len(pid) - 1: len(pid) - 1 + len(ct)].astype(mx.float32)
            y = mx.array(ct)
            ly = mx.take_along_axis(lg, y[:, None], axis=-1)
            r = mx.sum(lg > ly, axis=-1)          # 0-based rank of the emitted token under the reference
            mx.eval(r)
            ranks.append(np.array(r))
            del lg, ly, r
            mx.clear_cache()   # keep the MLX buffer cache from growing across many prefills
        r = np.concatenate(ranks)
        res[Path(f).name] = {"tokens": int(r.size), **{f"n_rank_ge_{t}": int((r >= t).sum()) for t in THRESH},
                             **{f"frac_rank_ge_{t}": float((r >= t).mean()) for t in THRESH}}
        print(Path(f).name, {k: v for k, v in res[Path(f).name].items() if k.startswith(("tokens", "n_rank"))}, flush=True)
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
