"""L2 (prefill consistency) with position matching, for the natural experiment.

For each commit's speculation-on run and the independent-seed speculation-off run of the same setting:
  * per-token TV against a clean teacher-forced prefill (tv_pre, stored by r2_mlx_mtp.py);
  * per prompt, the mean of (TV_on - TV_off) over the positions present in both sequences, and an exact-in-the-limit
    sign-flip permutation test over prompts (100k random sign vectors; prompts are the independent units);
  * a position-matched noise floor for every 'rejections so far' bin: the speculation-off mean TV at the positions of
    the speculation-on tokens in that bin (removes the position confound in Fig. 2).
Usage: python -I l2_matched.py  -> results/l2_matched.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0, 0, "0"), (1, 5, "1-5"), (6, 20, "6-20"), (21, 10**9, "21+")]
RUNS = {  # setting -> (spec-off file, {commit: spec-on file})
    "qwen_think_T0.6_p0.95_k20": ("r2x/D_head_qwen_think_T0.6_p0.95_k20_specoff.npz",
                                  {c: f"r2x/{c}_qwen_think_T0.6_p0.95_k20_specon.npz"
                                   for c in ("A_initial", "B_preResid", "C_preLpAcc", "D_head")}),
    "T1.0_plain": ("r2x/D_head_T1.0_plain_specoff.npz",
                   {c: f"r2/{c}_T1.0_plain_specon.npz" for c in ("A_initial", "B_preResid", "C_preLpAcc", "D_head")}),
}


def per_seq(d):
    out = {}
    for s in np.unique(d["seq"]):
        m = d["seq"] == s
        out[int(s)] = (d["step"][m], d["tv_pre"][m], d["n_rej"][m] if "n_rej" in d.files else np.zeros(m.sum()))
    return out


def prompt_of(path):
    meta = json.loads((ROOT / "results" / path.replace(".npz", ".seqs.json")).read_text())
    return {s["seq"]: s["prompt"] for s in meta["seqs"]}


def main(n_flip=100_000):
    R = ROOT / "results"
    res = {}
    rng = np.random.default_rng(0)
    for setting, (off_f, ons) in RUNS.items():
        if not (R / off_f).exists():
            print("missing", off_f); continue
        off = np.load(R / off_f)
        off_seq, off_pr = per_seq(off), prompt_of(off_f)
        off_by_prompt = {off_pr[s]: v for s, v in off_seq.items()}
        steps_all = off["step"]; tv_all = off["tv_pre"]
        floor_by_step = {int(t): float(tv_all[steps_all == t].mean()) for t in np.unique(steps_all)}
        sres = {"floor_overall": float(np.nanmean(tv_all))}
        for c, on_f in ons.items():
            if not (R / on_f).exists():
                print("missing", on_f); continue
            on = np.load(R / on_f)
            on_seq, on_pr = per_seq(on), prompt_of(on_f)
            diffs = []
            for s, (st, tv, _) in on_seq.items():
                p = on_pr[s]
                if p not in off_by_prompt:
                    continue
                st0, tv0, _ = off_by_prompt[p]
                common = np.intersect1d(st, st0)
                if common.size == 0:
                    continue
                a = tv[np.isin(st, common)]; b = tv0[np.isin(st0, common)]
                diffs.append(float(np.nanmean(a) - np.nanmean(b)))
            diffs = np.array(diffs)
            obs = diffs.mean()
            signs = rng.choice([-1.0, 1.0], size=(n_flip, diffs.size))
            null = (signs * diffs).mean(axis=1)
            p = float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-15)) / (n_flip + 1))
            bins = {}
            for lo, hi, name in BINS:
                m = (on["n_rej"] >= lo) & (on["n_rej"] <= hi)
                if not m.any():
                    continue
                matched = np.array([floor_by_step.get(int(t), np.nan) for t in on["step"][m]])
                bins[name] = {"n": int(m.sum()), "tv_on": float(np.nanmean(on["tv_pre"][m])),
                              "floor_position_matched": float(np.nanmean(matched))}
            sres[c] = {"prompts": int(diffs.size), "mean_diff_on_minus_off": float(obs), "signflip_p": p,
                       "tv_on": float(np.nanmean(on["tv_pre"])), "bins": bins}
            print(f"{setting:28s} {c:11s} TV on {sres[c]['tv_on']:.4f} vs floor {sres['floor_overall']:.4f}  "
                  f"per-prompt diff {obs:+.5f}  sign-flip p {p:.4g}  bins "
                  + " ".join(f"{k}:{v['tv_on']:.4f}/{v['floor_position_matched']:.4f}" for k, v in bins.items()), flush=True)
        res[setting] = sres
    (R / "l2_matched.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
