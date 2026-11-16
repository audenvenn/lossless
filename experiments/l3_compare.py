"""CPU-side L3 comparisons from cached reference scores (score_arms.py output).

Each comparison: exact permutation test, labels permuted within (prompt, sample index) strata, four statistics,
Westfall-Young min-p as the family p-value. Usage: python -I l3_compare.py PAIRS.json OUT.json
where PAIRS.json = [{"name": ..., "arm1": path.npz, "arm0": path.npz, "kind": "on_vs_off" | "AA" | ...}, ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lossless import twosample  # noqa: E402


def load(path):
    d = np.load(path)
    off = d["off"]
    vals = [d["z"][off[i]:off[i + 1]] for i in range(off.size - 1)]
    return vals, d["prompt"].astype(int), d["sample"].astype(int)


def compare(arm1, arm0, n_perm=2000, seed=0):
    v1, p1, s1 = load(arm1)
    v0, p0, s0 = load(arm0)
    vals = v1 + v0
    arm = np.r_[np.ones(len(v1), int), np.zeros(len(v0), int)]
    prompt = np.r_[p1, p0]
    strata = prompt * 1000 + np.r_[s1, s0]
    r = twosample.permutation_test(prompt, arm, vals, n_perm=n_perm, seed=seed, strata=strata)
    r["tokens_per_arm"] = [int(sum(x.size for x in v1)), int(sum(x.size for x in v0))]
    return r


def main():
    pairs = json.loads(Path(sys.argv[1]).read_text())
    out = {}
    for pr in pairs:
        r = compare(ROOT / pr["arm1"], ROOT / pr["arm0"], n_perm=pr.get("n_perm", 2000))
        out[pr["name"]] = r | {"kind": pr.get("kind", ""), "arm1": pr["arm1"], "arm0": pr["arm0"]}
        ps = r["p"]
        print(f"{pr['name']:48s} WY {r['p_minp_wy']:.4f}  mean {ps['mean']:.3f} ks {ps['ks']:.3f} "
              f"up {ps['upper_tail_frac']:.3f} lo {ps['lower_tail_frac']:.3f}  strata {r['n_strata']}  "
              f"tokens {r['tokens_per_arm']}", flush=True)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
