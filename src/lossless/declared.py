"""Exact declared distributions over tie classes, with screening of positions where exactness cannot be guaranteed.

L1 needs, at each position, the distribution the engine *declares* it samples from: the sampler transform applied
to the model's log-probabilities. Recomputing that distribution outside the engine has three pitfalls.

1. Ties at a truncation boundary. Engines differ in how they treat tokens tied with the boundary token:
   - 'count': keep exactly as many as the rule needs, chosen by some internal order (argpartition, a stable sort);
   - 'all':   keep every token whose value equals the boundary value (e.g. HF top-k, mlx-lm >= 0.31 top-p).
   We work with tie classes (tokens of equal value) and give each class its kept *count* under the declared rule, so
   the class mass is exact whichever tied members the engine actually keeps. Merging tied tokens into one category
   is a predictable coarsening, so the PIT over classes stays exact (paper, Lemma 3).
2. Numerical ambiguity. Engines evaluate cumulative sums and thresholds in their own precision. If a top-p crossing
   or a min-p threshold lies within `eps` of a class boundary, a recomputation may disagree with the engine. Such
   positions are screened out. The screen depends only on the distribution at that position, which is fixed before
   the token is drawn, so skipping them is a predictable selection and the remaining PIT values are still iid
   uniform (Lemma 2, Doob's optional skipping).
3. Partial information. With only the reported top-K plus the leftover tail mass, the declared distribution is
   computable exactly only if every operation can be resolved without knowing how the tail is composed. Positions
   where that fails are screened as 'insufficient'. Unreported tokens tied with the last reported value are
   handled through the boundary class.

Input values are T = 1 log-probabilities normalised over the full vocabulary (e.g. OpenAI `top_logprobs`, llama.cpp
`n_probs` without post-sampling). Output: a categorical distribution over classes (descending value), plus one
merged 'boundary + tail' category when the unreported mass stays in the support.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORDERS = {
    "hf": ("temp", "top_k", "top_p", "min_p"),
    "vllm": ("temp", "min_p", "top_k", "top_p"),
    "llamacpp": ("top_k", "top_p", "min_p", "temp"),
    "mlxlm": ("top_p", "min_p", "top_k", "temp"),
}

# Tie handling at the top-k / top-p boundary, as read from each engine's source (2026-11-16).
TIES = {
    "hf": {"top_k": "all", "top_p": "count"},          # TopKLogitsWarper keeps scores >= k-th; TopP removes by sorted position
    "vllm": {"top_k": "all", "top_p": "count"},
    "llamacpp": {"top_k": "count", "top_p": "count"},  # sort + resize
    "mlxlm": {"top_k": "count", "top_p": "all"},       # argpartition; mlx-lm >= 0.31 top-p keeps logprobs >= threshold
    "mlxlm_legacy": {"top_k": "count", "top_p": "count"},  # mlx-lm <= 0.30 / PR #990 clone: ascending cumsum by position
}


@dataclass
class Declared:
    status: str                  # "ok" | "ambiguous" | "insufficient" | "greedy"
    reason: str
    values: np.ndarray           # class values (T = 1 log-probs), descending
    mass: np.ndarray             # declared mass per category: one per class, then the merged tail category if any
    merged_tail: bool            # last category = boundary class (all members) + unreported tail
    boundary_value: float        # value of the last reported class (unreported ties of it map to that class)

    def category(self, token_value: float | None, reported: bool) -> int:
        """Category index of an emitted token, from its value; unreported tokens below the boundary map to the
        tail. Returns -1 if the token falls outside every category with positive declared mass."""
        nC = self.values.size
        if token_value is not None and (reported or token_value == self.boundary_value):
            hit = np.nonzero(self.values == token_value)[0]
            if hit.size:
                c = int(hit[0])
                if self.merged_tail and c == nC - 1:
                    return self.mass.size - 1
                return c
        if self.merged_tail:
            return self.mass.size - 1
        return -1


def _fail(status, reason):
    return Declared(status, reason, np.zeros(0), np.zeros(0), False, np.nan)


def declare(values, temp: float, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, semantics: str = "hf",
            ties: dict | None = None, has_tail: bool = True, eps: float = 1e-5) -> Declared:
    """Declared distribution over tie classes of the reported values (T = 1 log-probs, any order).

    has_tail: True if `values` are a top-K report (unreported tokens exist); False for a full vocabulary.
    eps: numerical tolerance of the engine's own arithmetic, absolute in probability for top-p crossings and in
         log-probability for min-p thresholds (raise it for engines that compute in bf16/fp16).
    """
    v = np.asarray(values, dtype=np.float64)
    order = ORDERS["mlxlm" if semantics == "mlxlm_legacy" else semantics]
    ties = dict(TIES.get(semantics, {"top_k": "count", "top_p": "count"}), **(ties or {}))
    cv, cn = np.unique(v, return_counts=True)
    cv, cn = cv[::-1].copy(), cn[::-1].astype(np.int64)          # classes, descending value
    nC = cv.size
    head = float(np.exp(cv) @ cn)
    tail_mass = max(0.0, 1.0 - head) if has_tail else 0.0
    if has_tail and head > 1.0 + 1e-4:
        return _fail("insufficient", "reported mass exceeds 1")
    if temp is None or temp <= 0:
        m = np.zeros(nC); m[0] = 1.0
        return Declared("greedy", "", cv, m, False, float(cv[-1]))
    # kept[c]: number of kept members of class c; -1 means 'all members' (count unknown for the boundary class
    # when has_tail). tail_kept: unreported tokens strictly below the boundary value are still in the support.
    kept = np.full(nC, -1, dtype=np.int64)
    if not has_tail:
        kept[:] = cn
    TAIL_EPS = 1e-12      # unreported mass below this is treated as zero (changes any distribution by <= 1e-12 TV)
    tail_kept = bool(has_tail and tail_mass > TAIL_EPS)
    tail_partial = False      # top-p crossed inside the unreported tail: some unknown part of it is kept
    scale = 1.0
    status, reason = "ok", ""

    def n_kept(c):
        return cn[c] if kept[c] < 0 else kept[c]

    def known_total():
        """Total current weight of the support, or None if it depends on the unreported tokens' composition."""
        w = np.exp(cv * scale)
        tot = 0.0
        for c in range(nC):
            if kept[c] == 0:
                continue
            if kept[c] < 0 and c == nC - 1 and has_tail:       # boundary class, all members (unknown count)
                if scale != 1.0 or not tail_kept:
                    return None
                tot += w[c] * cn[c]                            # visible members; unseen ones are inside tail_mass
                continue
            tot += w[c] * n_kept(c)
        if tail_kept:
            if scale != 1.0 or tail_partial:
                return None
            tot += tail_mass
        return tot

    for op in order:
        if op == "temp":
            scale = 1.0 / temp
        elif op == "top_k":
            if not top_k or top_k <= 0:
                continue
            rem = int(top_k)
            for c in range(nC):
                if kept[c] == 0:
                    continue
                if rem <= 0:
                    kept[c] = 0
                    continue
                unknown = kept[c] < 0 and c == nC - 1 and has_tail   # boundary class, total size unknown
                avail = cn[c] if unknown else n_kept(c)               # members known to be in the support
                if rem > avail or (unknown and ties["top_k"] == "all"):
                    if unknown:
                        return _fail("insufficient", "top-k needs the unreported tokens")
                    kept[c] = avail
                    rem -= avail
                else:                                                  # the cut falls inside class c
                    kept[c] = avail if ties["top_k"] == "all" else rem
                    rem = 0
            if rem > 0 and tail_kept:
                return _fail("insufficient", "top-k reaches into the unreported tail")
            if rem <= 0:
                tail_kept = False
                tail_partial = False
        elif op == "top_p":
            if top_p is None or top_p >= 1.0 or top_p <= 0.0:
                continue
            tot = known_total()
            if tot is None:
                return _fail("insufficient", "top-p normalisation needs the tail composition")
            w = np.exp(cv * scale) / tot
            cum = 0.0
            crossed = False
            for c in range(nC):
                if kept[c] == 0:
                    continue
                if crossed:
                    kept[c] = 0
                    continue
                boundary_unknown = kept[c] < 0 and c == nC - 1 and has_tail
                nk = cn[c] if boundary_unknown else n_kept(c)    # visible count for the boundary class
                cls = w[c] * nk
                if cum + cls < top_p - eps and not boundary_unknown:
                    cum += cls
                    continue
                if cum + cls < top_p - eps and boundary_unknown:
                    if ties["top_p"] == "all" or not tail_kept:
                        return _fail("insufficient", "top-p crossing lies beyond the reported tokens")
                    tail_partial = True               # whole boundary class + part of the tail; a later cut may resolve
                    crossed = True
                    continue
                # crossing inside class c: smallest j with cum + j*w >= top_p
                j = int(np.ceil((top_p - cum) / w[c] - 1e-12))
                j = min(max(j, 1), nk)
                lo, hi = cum + (j - 1) * w[c], cum + j * w[c]
                if abs(hi - top_p) < eps or abs(lo - top_p) < eps:
                    status, reason = "ambiguous", "top-p crossing within eps of a token boundary"
                if ties["top_p"] == "all":
                    if boundary_unknown:
                        return _fail("insufficient", "top-p keeps all ties of a partially reported class")
                    kept[c] = n_kept(c)
                else:
                    kept[c] = j
                crossed = True
            if not crossed and tail_kept:
                tail_partial = True                       # crossing inside the tail; a later top-k/min-p may resolve
            elif crossed and not tail_partial:
                tail_kept = False
        elif op == "min_p":
            if not min_p or min_p <= 0.0:
                continue
            live = [c for c in range(nC) if kept[c] != 0]
            if not live:
                return _fail("insufficient", "empty support")
            lthr = cv[live[0]] * scale + np.log(min_p)            # top live class is always reported
            for c in live:
                lw = cv[c] * scale
                if abs(lw - lthr) < eps:
                    status, reason = "ambiguous", "min-p threshold within eps of a class value"
                if lw < lthr:
                    kept[c] = 0
            if tail_kept:
                if cv[-1] * scale < lthr - eps:
                    tail_kept = False                              # every unreported token is below the threshold
                    tail_partial = False
                else:
                    return _fail("insufficient", "min-p threshold may admit unreported tokens")

    # final categorical over the support
    if tail_partial and (tail_kept or kept[nC - 1] < 0):
        return _fail("insufficient", "top-p keeps an undetermined part of the unreported tail")
    w = np.exp(cv * scale)
    unknown_boundary = has_tail and kept[nC - 1] < 0       # boundary class kept whole, size unknown (=> tail kept)
    if unknown_boundary and not tail_kept and tail_mass <= TAIL_EPS:
        kept[nC - 1] = cn[nC - 1]                           # nothing unreported carries mass: class is complete
        unknown_boundary = False
    merged = unknown_boundary or tail_kept
    if merged and scale != 1.0:
        return _fail("insufficient", "tempered mass of unreported tokens")
    if merged and not (unknown_boundary and tail_kept):
        return _fail("insufficient", "inconsistent boundary state")
    n_cls = nC - 1 if merged else nC
    mass = np.array([w[c] * n_kept(c) if kept[c] != 0 else 0.0 for c in range(n_cls)])
    if merged:                    # all members of the boundary class + every unreported token, at T = 1
        mass = np.append(mass, w[nC - 1] * cn[nC - 1] + tail_mass)
    s = mass.sum()
    if not s > 0:
        return _fail("insufficient", "empty support")
    return Declared(status, reason, cv, mass / s, merged, float(cv[-1]))


def pit(decl: Declared, token_value: float | None, reported: bool, u: float):
    """Randomised PIT over the declared categories. Returns (z, surprisal, entropy, var_surprisal, violation)."""
    c = decl.category(token_value, reported)
    m = decl.mass
    lm = np.log(np.maximum(m, 1e-300))
    ent = float(-(m * lm).sum()); var = max(0.0, float((m * lm * lm).sum()) - ent * ent)
    if c < 0 or m[c] <= 0:
        return 1.0, 700.0, ent, var, True
    return float(m[:c].sum() + u * m[c]), float(-lm[c]), ent, var, False
