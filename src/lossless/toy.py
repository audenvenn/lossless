"""A toy order-2 language model with exact conditionals, plus a vectorized reference speculative sampler
and realistic bug variants. Used to validate the PIT tests (null exactness, power per bug class) before
touching real engines.

Context = (prev1, prev2): the last and second-to-last tokens. All conditional distributions are
precomputed tables of shape (V, V, V): table[prev1, prev2] is a distribution over the next token.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import samplers


@dataclass
class ToyLM:
    target_logits: np.ndarray  # (V, V, V)
    draft_logits: np.ndarray   # (V, V, V)

    @property
    def V(self) -> int:
        return self.target_logits.shape[-1]

    @staticmethod
    def random(V: int = 48, draft_noise: float = 0.8, seed: int = 0) -> "ToyLM":
        """Rows with LLM-like entropy spread: many confident contexts, some flat ones."""
        rng = np.random.default_rng(seed)
        a = rng.gumbel(size=(V, V))                     # depends on prev1
        b = rng.gumbel(size=(V, V))                     # depends on prev2
        scale = np.exp(rng.uniform(np.log(0.6), np.log(6.0), size=(V, V)))  # per-context sharpness
        tl = scale[:, :, None] * (a[:, None, :] + 0.5 * b[None, :, :])
        dl = tl + draft_noise * rng.standard_normal(size=(V, V, V)) * (1 + 0.3 * scale[:, :, None])
        return ToyLM(tl, dl)

    def tables(self, temp=1.0, top_k=0, top_p=1.0, min_p=0.0, semantics="hf"):
        """Transformed (claimed) target and draft distributions for every context."""
        V = self.V
        P = np.empty((V, V, V))
        Q = np.empty((V, V, V))
        for i in range(V):
            for j in range(V):
                P[i, j] = samplers.transform(self.target_logits[i, j], temp, top_k, top_p, min_p, semantics)
                Q[i, j] = samplers.transform(self.draft_logits[i, j], temp, top_k, top_p, min_p, semantics)
        return P, Q


def pit_tables(P: np.ndarray):
    """For each context: F(y-) and p(y) with tokens ordered by decreasing probability (stable)."""
    V = P.shape[-1]
    order = np.argsort(-P, axis=-1, kind="stable")
    sorted_p = np.take_along_axis(P, order, axis=-1)
    cum_before = np.cumsum(sorted_p, axis=-1) - sorted_p
    fminus = np.empty_like(P)
    np.put_along_axis(fminus, order, cum_before, axis=-1)
    return fminus


def _sample_rows(dist_rows: np.ndarray, rng) -> np.ndarray:
    """Sample one token from each row of a (B, V) array of (possibly unnormalized, nonnegative) weights."""
    w = np.clip(dist_rows, 0.0, None)
    cdf = np.cumsum(w, axis=1)
    tot = cdf[:, -1:]
    u = rng.random((w.shape[0], 1)) * tot
    idx = (cdf < u).sum(axis=1)
    return np.minimum(idx, w.shape[1] - 1)


VARIANTS = (
    "correct",
    "resample_target",        # on rejection sample from p instead of the residual (p - q)+
    "argmax_on_reject",       # on rejection emit argmax p (mlx-lm PR #990 pre-fix bug class)
    "bonus_from_draft",       # all K accepted: bonus token drawn from the draft instead of the target
    "accept_untransformed",   # acceptance ratio uses T=1/untruncated p, q; sampling/residual use transformed
    "residual_untransformed", # residual built from T=1/untruncated p, q
    "target_kv_stale",        # after a rejection the target verifies the next position with a stale context
)


def simulate_spec(model: ToyLM, P, Q, P1, Q1, K: int, n_tokens: int, variant: str, rng, B: int = 2000,
                  return_chain: bool = False, start=None, stats: dict | None = None):
    """Run vectorized speculative decoding on B chains until >= n_tokens are emitted.

    P, Q: claimed (transformed) target/draft tables; P1, Q1: untransformed (T=1) tables for bug variants.
    Returns arrays (ctx1, ctx2, token) for every emitted token, where (ctx1, ctx2) is the TRUE context
    (the emitted history), so the claimed distribution is P[ctx1, ctx2].
    """
    V = model.V
    s1 = rng.integers(V, size=B) if start is None else np.full(B, start[0])
    s2 = rng.integers(V, size=B) if start is None else np.full(B, start[1])
    stale = np.full(B, -1)  # for target_kv_stale: token the target wrongly sees as prev1 at next position
    out_c1, out_c2, out_y, out_ch = [], [], [], []
    emitted = 0
    ar = np.arange(B)
    while emitted < n_tokens:
        # draft K tokens autoregressively from the draft model (correct contexts)
        d = np.empty((B, K), dtype=np.int64)
        c1 = np.empty((B, K + 1), dtype=np.int64)
        c2 = np.empty((B, K + 1), dtype=np.int64)
        c1[:, 0], c2[:, 0] = s1, s2
        for k in range(K):
            d[:, k] = _sample_rows(Q[c1[:, k], c2[:, k]], rng)
            c1[:, k + 1], c2[:, k + 1] = d[:, k], c1[:, k]
        # target contexts used for verification (stale bug only touches the first position)
        t1 = c1.copy()
        if variant == "target_kv_stale":
            has = stale >= 0
            t1[has, 0] = stale[has]
        pk = P[t1[:, :K], c2[:, :K], d]
        qk = Q[c1[:, :K], c2[:, :K], d]
        if variant == "accept_untransformed":
            ratio = P1[t1[:, :K], c2[:, :K], d] / np.maximum(Q1[c1[:, :K], c2[:, :K], d], 1e-300)
        else:
            ratio = pk / np.maximum(qk, 1e-300)
        acc = rng.random((B, K)) < np.minimum(1.0, ratio)
        rej_any = ~acc.all(axis=1)
        j = np.where(rej_any, np.argmin(acc, axis=1), K)  # number of accepted drafts
        # final token
        fin = np.empty(B, dtype=np.int64)
        rej = j < K
        jr = j[rej]
        rc1, rc2 = t1[rej, jr], c2[rej, jr]
        qc1 = c1[rej, jr]
        if variant == "resample_target":
            fin[rej] = _sample_rows(P[rc1, rc2], rng)
        elif variant == "argmax_on_reject":
            fin[rej] = np.argmax(P[rc1, rc2], axis=1)
        elif variant == "residual_untransformed":
            res = P1[rc1, rc2] - Q1[qc1, rc2]
            # fall back to target if residual is empty (keeps sampler defined)
            empty = np.clip(res, 0, None).sum(axis=1) <= 0
            res[empty] = P1[rc1[empty], rc2[empty]]
            # engines would then apply the declared transform's support; emulate by masking to P's support
            res = np.where(P[rc1, rc2] > 0, res, 0.0)
            empty2 = np.clip(res, 0, None).sum(axis=1) <= 0
            res[empty2] = P[rc1[empty2], rc2[empty2]]
            fin[rej] = _sample_rows(res, rng)
        else:
            res = P[rc1, rc2] - Q[qc1, rc2]
            empty = np.clip(res, 0, None).sum(axis=1) <= 1e-15
            res[empty] = P[rc1[empty], rc2[empty]]
            fin[rej] = _sample_rows(res, rng)
        allacc = ~rej
        bc1, bc2 = c1[allacc, K], c2[allacc, K]
        if variant == "bonus_from_draft":
            fin[allacc] = _sample_rows(Q[bc1, bc2], rng)
        else:
            fin[allacc] = _sample_rows(P[bc1, bc2], rng)
        # collect emitted tokens with their TRUE contexts
        for b_k in range(K):
            m_acc = j > b_k  # draft token b_k is emitted iff it was accepted
            out_c1.append(c1[m_acc, b_k]); out_c2.append(c2[m_acc, b_k]); out_y.append(d[m_acc, b_k])
            out_ch.append(ar[m_acc])
        # final token context: (c1[j], c2[j]) true context at position j
        out_c1.append(c1[ar, j]); out_c2.append(c2[ar, j]); out_y.append(fin); out_ch.append(ar)
        emitted += int(j.sum() + B)
        if stats is not None:
            stats["accepted"] = stats.get("accepted", 0) + int(j.sum())
            stats["proposed"] = stats.get("proposed", 0) + int(np.minimum(j + 1, K).sum())
        # new state
        last = fin
        prev = np.where(j > 0, d[ar, np.maximum(j - 1, 0)], s1)
        stale = np.where(rej, d[ar, np.minimum(j, K - 1)], -1)
        s1, s2 = last, prev
    if return_chain:
        return np.concatenate(out_c1), np.concatenate(out_c2), np.concatenate(out_y), np.concatenate(out_ch)
    return np.concatenate(out_c1), np.concatenate(out_c2), np.concatenate(out_y)


def simulate_plain(P_true, n_tokens: int, rng, B: int = 2000, return_chain: bool = False):
    """Non-speculative sampling from P_true (the distribution actually used); returns contexts and tokens."""
    V = P_true.shape[-1]
    s1 = rng.integers(V, size=B)
    s2 = rng.integers(V, size=B)
    c1s, c2s, ys = [], [], []
    for _ in range(int(np.ceil(n_tokens / B))):
        y = _sample_rows(P_true[s1, s2], rng)
        c1s.append(s1); c2s.append(s2); ys.append(y)
        s1, s2 = y, s1
    out = (np.concatenate(c1s), np.concatenate(c2s), np.concatenate(ys))
    if return_chain:
        return out + (np.tile(np.arange(B), len(ys)),)
    return out


def pit_values(P_claimed, c1, c2, y, rng):
    fm = pit_tables(P_claimed)
    u = rng.random(y.size)
    return fm[c1, c2, y] + u * P_claimed[c1, c2, y]


def surprisal_from_tables(P_claimed, c1, c2, y):
    rows = P_claimed[c1, c2]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(rows > 0, np.log(rows), 0.0)
    ent = -(rows * logp).sum(axis=1)
    var = np.clip((rows * logp**2).sum(axis=1) - ent**2, 0.0, None)
    s = -logp[np.arange(y.size), y]
    return s, ent, var
