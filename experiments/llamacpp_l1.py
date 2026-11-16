"""L1 for llama.cpp from its own per-token post-sampling distributions (request field post_sampling_probs=true).

Each emitted token entry carries `id`, `prob` (its probability under the distribution after the sampler chain) and
`top_probs` (the top-N of that distribution). The randomized PIT is computed over the reported top-N plus one tail
bucket (Lemma 3), with tied probabilities merged into one class. Under H0 the Z values are iid U(0,1).
Usage: python -I llamacpp_l1.py FILE.jsonl [FILE2.jsonl ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lossless import pit  # noqa: E402


def analyze(path: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    zs, s_, e_, v_ = [], [], [], []
    n_tok = viol = not_listed = 0
    roles = []  # position index within reply (prefix-measurable stratum)
    for line in open(path):
        rec = json.loads(line)
        for t, e in enumerate((rec.get("logprobs") or {}).get("content") or []):
            tops = e.get("top_probs") or []
            if not tops:
                continue
            n_tok += 1
            probs = np.array([x["prob"] for x in tops], dtype=np.float64)
            ids = [x["id"] for x in tops]
            head = probs.sum()
            tail = max(0.0, 1.0 - head)
            p_y = float(e["prob"])
            # tie classes over the reported head (ordered by descending prob)
            order = np.argsort(-probs, kind="stable")
            probs_s = probs[order]; ids_s = [ids[i] for i in order]
            vals, inv = np.unique(-probs_s, return_inverse=True)   # classes in descending-prob order
            cls_mass = np.bincount(inv, weights=probs_s)
            cls_cum = np.concatenate([[0.0], np.cumsum(cls_mass)])
            if e["id"] in ids_s:
                c = inv[ids_s.index(e["id"])]
                zs.append(cls_cum[c] + rng.random() * cls_mass[c]); s_.append(-np.log(cls_mass[c]))
            elif tail > 1e-12:
                not_listed += 1
                zs.append(cls_cum[-1] + rng.random() * tail); s_.append(-np.log(tail))
            else:
                viol += 1; zs.append(1.0); s_.append(700.0)
            cats = np.concatenate([cls_mass, [tail]]) if tail > 1e-12 else cls_mass
            cats = cats / cats.sum()
            lq = np.log(np.maximum(cats, 1e-300))
            ent = float(-(cats * lq).sum()); m2 = float((cats * lq ** 2).sum())
            e_.append(ent); v_.append(max(0.0, m2 - ent ** 2)); roles.append(t)
    rep = pit.uniformity_tests(np.array(zs), surprisal=(np.array(s_), np.array(e_), np.array(v_)))
    return rep.as_dict() | {"reject_bonf_0.01": bool(pit.bonferroni_reject(rep, 0.01)), "tokens": n_tok,
                            "support_violations": viol, "emitted_outside_reported_top": not_listed}


if __name__ == "__main__":
    out = {}
    for f in sys.argv[1:]:
        r = analyze(f)
        out[Path(f).name] = r
        print(f"{Path(f).name:52s} n={r['tokens']:6d} KS p={r['ks_p']:.3f} meanz={r['mean_z']:.4f} "
              f"surprisal z={r['surprisal_z']:+.2f} viol={r['support_violations']} outside_top={r['emitted_outside_reported_top']} "
              f"REJECT={r['reject_bonf_0.01']}")
    (ROOT / "results" / "l3" / "llamacpp_l1.json").write_text(json.dumps(out, indent=1))
