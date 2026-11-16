"""R3: greedy identity of speculative runs vs D-head spec-off (temp 0). Per prompt: identical? first divergence."""
import json
import sys
from pathlib import Path

import numpy as np

R3 = Path(__file__).resolve().parents[1] / "results" / "r3"
ref = {s["prompt"]: s["tokens"] for s in json.loads((R3 / "D_head_greedy_specoff.seqs.json").read_text())["seqs"]}
out = {}
for f in sorted(R3.glob("*_greedy_specon.seqs.json")):
    seqs = {s["prompt"]: s["tokens"] for s in json.loads(f.read_text())["seqs"]}
    same, firsts = 0, []
    for p, toks in seqs.items():
        r = ref[p]
        if toks == r:
            same += 1
            continue
        n = next((i for i, (x, y) in enumerate(zip(toks, r)) if x != y), min(len(toks), len(r)))
        firsts.append(n)
    tag = f.name.replace(".seqs.json", "")
    out[tag] = {"prompts": len(seqs), "identical": same, "diverged": len(firsts),
                "median_first_divergence": float(np.median(firsts)) if firsts else None,
                "first_divergences": sorted(firsts)}
    print(f"{tag:40s} identical {same}/{len(seqs)}; diverged {len(firsts)}; median first divergence "
          f"{out[tag]['median_first_divergence']}; {sorted(firsts)[:12]}")
(R3 / "summary.json").write_text(json.dumps(out, indent=1))
