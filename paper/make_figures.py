"""Paper figures (static, print, light surface). Palette: reference categorical slots 1-3 (validated all-pairs,
light mode: blue #2a78d6, orange #eb6834, aqua #1baf7a; aqua <3:1 contrast -> always direct-labelled)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"  # white for print
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "font.size": 8.5, "axes.titlesize": 8.5,
    "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 4,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_power():
    d = json.loads((ROOT / "results" / "synthetic_power.json").read_text())
    ns = d["NS"]
    deploy, t1 = d["settings"]["T0.7_k20_p0.95"], d["settings"]["T1.0"]
    names = {"resample_target": "resample on reject", "argmax_on_reject": "argmax on reject",
             "bonus_from_draft": "bonus from draft", "accept_untransformed": "untransformed accept",
             "residual_untransformed": "untransformed residual", "target_kv_stale": "stale state*",
             "temp_plus5pct": "temperature +5%", "temp_last_vs_first": "temp-last vs temp-first",
             "silent_min_p_0.05": "silent min-p 0.05", "top_p_exclusive": "exclusive top-p"}
    fig, axes = plt.subplots(2, 5, figsize=(9.2, 3.6), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, names.items()):
        ax.plot(ns, deploy[key]["power"], color=BLUE, marker="o", label="T 0.7, top-k 20, top-p 0.95")
        ax.plot(ns, t1[key]["power"], color=ORANGE, marker="o", label="T 1, no truncation")
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03); ax.set_title(title)
        ax.set_xticks([300, 3000, 30000]); ax.set_xticklabels(["300", "3k", "30k"])
    for ax in axes[:, 0]:
        ax.set_ylabel("power (α = 0.01)")
    for ax in axes[1]:
        ax.set_xlabel("emitted tokens")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.text(0.01, 0.005, "*scored against the true conditionals (a fixed exact reference); L1 against an engine's own reported distribution cannot see state errors", fontsize=6.5, color=INK2)
    save(fig, "fig1_power")


def fig_dose():
    """Per-token TV vs a clean prefill by rejections so far, with a POSITION-MATCHED noise floor per bin
    (speculation-off mean TV at the positions of the speculation-on tokens in that bin; results/l2_matched.json).
    Thinking: independent-seed R2x runs; T = 1: R2 speculation-on runs vs the R2x seed-205 control."""
    l2 = json.loads((ROOT / "results" / "l2_matched.json").read_text())
    names = ["0", "1-5", "6-20", "21+"]
    labels = ["0", "1–5", "6–20", "21+"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, (setting, title) in zip(axes, (("T1.0_plain", "T 1"), ("qwen_think_T0.6_p0.95_k20", "thinking (T 0.6, k 20, p 0.95)"))):
        d = l2[setting]
        for tag, color, name in (("A_initial", ORANGE, "commit A (state contamination)"), ("D_head", BLUE, "commit D (fixed)")):
            ks = [k for k in names if k in d[tag]["bins"]]
            ax.plot([names.index(k) for k in ks], [d[tag]["bins"][k]["tv_on"] for k in ks], color=color, marker="o", label=name)
        ks = [k for k in names if k in d["D_head"]["bins"]]
        ax.plot([names.index(k) for k in ks], [d["D_head"]["bins"][k]["floor_position_matched"] for k in ks], color=INK2,
                linestyle="--", linewidth=0.9, marker="o", markersize=2.5, markerfacecolor=SURFACE,
                label="noise floor, position-matched (speculation off)")
        ax.set_yscale("log"); ax.set_xticks(range(len(names))); ax.set_xticklabels(labels)
        ax.set_xlabel("rejections so far"); ax.set_title(title)
    axes[0].set_ylabel("per-token TV vs clean prefill")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig2_dose_response")


def _boot_share(rank, off, t, reps=2000, seed=0):
    """Share of tokens with reference rank >= t, with a sequence-bootstrap 95% interval."""
    rng = np.random.default_rng(seed)
    hits = np.array([(rank[off[i]:off[i + 1]] >= t).sum() for i in range(off.size - 1)], dtype=float)
    lens = np.diff(off).astype(float)
    idx = rng.integers(0, hits.size, size=(reps, hits.size))
    boots = hits[idx].sum(1) / lens[idx].sum(1)
    return hits.sum() / lens.sum(), np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def fig_ranks():
    z = ROOT / "results" / "zref" / "mlx4"
    rows = [("kapitsa", "spec off", "l3__kapitsa_T1.0_plain_armoff.npz", True),
            ("kapitsa", "MTP", "l3__kapitsa_T1.0_plain_armmtp.npz", False),
            ("MTPLX", "spec off", "l3__mtplx_T1.0_plain_armoff.npz", True),
            ("MTPLX", "MTP", "l3__mtplx_T1.0_plain_armmtp.npz", False),
            ("raceline 0.8.0", "spec off", "l3__raceline_T1.0_plain_armoff2.npz", True),
            ("raceline 0.8.0", "MTP", "l3__raceline_T1.0_plain_armon.npz", False),
            ("raceline 0.8.1", "MTP", "l3__raceline081_T1.0_plain_armon.npz", False),
            ("llama.cpp", "spec off", "l3__llamacpp_T1.0_plain_armoff.npz", True),
            ("llama.cpp", "draft-MTP", "l3__llamacpp_T1.0_plain_armmtp.npz", False)]
    rows = [r for r in rows if (z / r[2]).exists()]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, t in zip(axes, (32, 64)):
        for y, (eng, arm, f, filled) in enumerate(rows):
            d = np.load(z / f)
            v, lo, hi = (100 * x for x in _boot_share(d["rank"], d["off"], t))
            out[f"{eng} {arm} rank>={t}"] = [v, lo, hi]
            ax.plot([lo, hi], [y, y], color=INK2, linewidth=1.0, zorder=1)
            ax.plot(v, y, marker="o", markersize=5.5, color=BLUE, markerfacecolor=BLUE if filled else SURFACE,
                    markeredgewidth=1.4, linestyle="none", zorder=2)
            ax.text(hi + 0.06, y, f"{v:.2f}%", va="center", fontsize=7, color=INK2)
        ax.set_title(f"beyond reference rank {t}"); ax.set_xlabel("% of emitted tokens")
        ax.set_xlim(0, 2.4); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([f"{e} · {a}" for e, a, *_ in rows])
    axes[0].invert_yaxis()
    fig.suptitle("T = 1, top_k = 0 requested (no truncation). Filled = speculation off, open = on;\n"
                 "bars = 95% sequence-bootstrap intervals; ranks under the MLX 4-bit reference", fontsize=8, y=1.04)
    fig.tight_layout()
    save(fig, "fig3_rank_support")
    (ROOT / "results" / "fig3_rank_shares.json").write_text(json.dumps(out, indent=1))


def fig_tail():
    d = np.load(ROOT / "results" / "rl_logits.npz")
    vals, lse = d["logits"].astype(np.float64), d["lse"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, color, name in ((32, BLUE, "beyond rank 32 (kapitsa cap)"), (1024, ORANGE, "beyond rank 1024 (raceline 0.8.0)")):
        t = 1 - np.exp(np.logaddexp.reduce(vals[:, :k], axis=1) - lse)
        xs = np.sort(np.clip(t, 1e-9, 1))
        ccdf = 1 - np.arange(1, xs.size + 1) / xs.size
        ax.step(xs, ccdf, where="post", color=color, label=name)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-6, 1); ax.set_ylim(1e-3, 1.05)
    ax.set_xlabel("probability mass removed at a position (T = 1)"); ax.set_ylabel("fraction of positions exceeding")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig4_tail_mass")


if __name__ == "__main__":
    fig_power(); fig_dose(); fig_ranks(); fig_tail()
    print("figures written to", OUT)
