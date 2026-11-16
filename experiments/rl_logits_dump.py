"""Dump realistic next-token logits (Qwen3.5-4B, MLX 4-bit, teacher-forced) at many positions of the R1 sequences,
for an exact, server-free test of raceline's numpy sampler. Stores per position: top-4096 ids/logits (descending),
the full-vocabulary logsumexp at T=1, and the probability mass beyond rank 1024 at several temperatures.

Run under the GPU lock (prefill only, ~2 min, ~4 GB).
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
from r1_mlxlm_null import SETTINGS  # noqa: E402

K = 4096
STRIDE = 8
TEMPS = (1.0, 0.7, 0.6)


def main():
    from mlx_lm import load
    model, tok = load(paths.QWEN_MLX4)
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text())
    out = {"ids": [], "logits": [], "lse": [], "tail1024": [], "setting": [], "prompt": [], "pos": []}
    for sname in ("T1.0_plain", "qwen_think_T0.6_p0.95_k20"):
        think = SETTINGS[sname][4]
        r = np.load(ROOT / "results" / "r1" / f"r1_{sname}.npz")
        for pi in np.unique(r["prompt"]):
            toks = r["token"][r["prompt"] == pi].tolist()
            msgs = [{"role": "user", "content": prompts[int(pi)]}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=think)
            logits = model(mx.array(list(ids) + toks)[None])[0].astype(mx.float32)
            for t in range(0, len(toks), STRIDE):
                row = logits[len(ids) - 1 + t]
                order = mx.argsort(-row)[:K]
                vals = row[order]
                tails = []
                for T in TEMPS:
                    z = row / T
                    lse = mx.logsumexp(z)
                    top = mx.logsumexp(vals[:1024] / T)
                    tails.append(1.0 - mx.exp(top - lse))
                lse1 = mx.logsumexp(row)
                mx.eval(order, vals, lse1, *tails)
                out["ids"].append(np.array(order, dtype=np.int32)); out["logits"].append(np.array(vals, dtype=np.float32))
                out["lse"].append(float(lse1)); out["tail1024"].append([float(x) for x in tails])
                out["setting"].append(sname); out["prompt"].append(int(pi)); out["pos"].append(t)
            del logits
        print(sname, "positions so far", len(out["lse"]), flush=True)
    np.savez_compressed(ROOT / "results" / "rl_logits.npz", ids=np.stack(out["ids"]), logits=np.stack(out["logits"]),
                        lse=np.array(out["lse"]), tail1024=np.array(out["tail1024"]), temps=np.array(TEMPS),
                        setting=np.array(out["setting"]), prompt=np.array(out["prompt"]), pos=np.array(out["pos"]))
    t = np.array(out["tail1024"])
    for j, T in enumerate(TEMPS):
        print(f"T={T}: mass beyond rank 1024: mean {t[:, j].mean():.2e}, p50 {np.median(t[:, j]):.2e}, "
              f"p99 {np.quantile(t[:, j], 0.99):.2e}, max {t[:, j].max():.2e}")


if __name__ == "__main__":
    main()
