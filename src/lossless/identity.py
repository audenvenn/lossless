"""Same-seed identity check for engines that claim speculation 'changes no output' / is 'bit-for-bit'.

Input: two L3 reply files collected with identical per-request seeds (arm on / arm off). For each (prompt,
sample) pair, compare the full completion (reasoning + content). Report the fraction identical and, for
mismatches, the first divergence position in tokens (reference tokenizer), so near-tie numerics (late, rare
divergences) can be told apart from algorithmic differences (early, frequent divergences).

Usage: lossless identity ON.jsonl OFF.jsonl --tokenizer PATH [--out x.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    out = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        out[(r["prompt"], r["sample"])] = r
    return out


def text(r):
    return (r.get("reasoning") or "") + "\x00" + (r.get("content") or "")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lossless identity")
    ap.add_argument("on", type=Path)
    ap.add_argument("off", type=Path)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer name or path (e.g. Qwen/Qwen3.5-4B)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    A, B = load(a.on), load(a.off)
    keys = sorted(set(A) & set(B))
    same, firsts, seed_mismatch = 0, [], 0
    for k in keys:
        if A[k].get("seed") != B[k].get("seed"):
            seed_mismatch += 1
            continue
        ta, tb = text(A[k]), text(B[k])
        if ta == tb:
            same += 1
            continue
        ia, ib = tok.encode(ta, add_special_tokens=False), tok.encode(tb, add_special_tokens=False)
        n = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
        firsts.append(n)
    res = {"pairs": len(keys), "identical": same, "frac_identical": same / max(1, len(keys) - seed_mismatch),
           "seed_mismatch": seed_mismatch, "first_divergence_tokens": sorted(firsts),
           "median_first_divergence": (sorted(firsts)[len(firsts) // 2] if firsts else None)}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
