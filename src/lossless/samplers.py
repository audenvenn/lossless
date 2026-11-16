"""Sampler transforms: map logits to the distribution an engine claims to sample from.

Engines disagree on the order of operations for the same parameters (verified from source, 2026-11-16):

  hf        temperature -> top_k -> top_p -> min_p        (transformers.generation.utils)
  vllm      temperature -> min_p -> top_k/top_p          (vllm/v1/worker/gpu/sample/sampler.py)
  llamacpp  top_k -> top_p -> min_p -> temperature       (common/common.h default chain; truncation on T=1 probs)
  mlxlm     top_p -> min_p -> top_k -> temperature       (mlx_lm/sample_utils.make_sampler; truncation on T=1 logprobs)

Each function takes a 1-D float64 array of logits and returns a probability vector (zeros outside the
support). Ties are broken by token id (lower id first), consistently with a stable descending sort.
"""
from __future__ import annotations

import numpy as np

NEG_INF = -np.inf


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z[np.isfinite(z)]) if np.any(np.isfinite(z)) else 0.0
    e = np.where(np.isfinite(z), np.exp(z - m), 0.0)
    s = e.sum()
    return e / s


def _order_desc(z: np.ndarray) -> np.ndarray:
    # stable: equal values keep ascending token id
    return np.argsort(-z, kind="stable")


def mask_top_k(z: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 0 or k >= z.size:
        return z
    out = np.full_like(z, NEG_INF)
    idx = _order_desc(z)[:k]
    out[idx] = z[idx]
    return out


def mask_top_p(z: np.ndarray, p: float, min_keep: int = 1) -> np.ndarray:
    """Keep the smallest prefix (by descending prob) whose mass >= p (the token that crosses is kept)."""
    if p is None or p >= 1.0 or p <= 0.0:
        return z
    probs = softmax(z)
    idx = _order_desc(z)
    cum = np.cumsum(probs[idx])
    n_keep = int(np.searchsorted(cum, p - 1e-12, side="left")) + 1
    n_keep = max(n_keep, min_keep)
    out = np.full_like(z, NEG_INF)
    keep = idx[:n_keep]
    out[keep] = z[keep]
    return out


def mask_min_p(z: np.ndarray, min_p: float, min_keep: int = 1) -> np.ndarray:
    """Remove tokens whose prob < min_p * max prob (relative threshold; invariant to renormalization)."""
    if min_p is None or min_p <= 0.0:
        return z
    probs = softmax(z)
    thr = min_p * probs.max()
    keep = probs >= thr
    if keep.sum() < min_keep:
        idx = _order_desc(z)[:min_keep]
        keep = np.zeros_like(keep)
        keep[idx] = True
    return np.where(keep, z, NEG_INF)


def transform(logits: np.ndarray, temp: float = 1.0, top_k: int = 0, top_p: float = 1.0,
              min_p: float = 0.0, semantics: str = "hf") -> np.ndarray:
    """Distribution claimed by an engine with the given parameters and order-of-operations semantics."""
    z = np.asarray(logits, dtype=np.float64)
    if temp is None or temp <= 0:
        out = np.zeros_like(z)
        out[_order_desc(z)[0]] = 1.0
        return out
    if semantics == "hf":
        z = z / temp
        z = mask_top_k(z, top_k)
        z = mask_top_p(z, top_p)
        z = mask_min_p(z, min_p)
        return softmax(z)
    if semantics == "vllm":
        z = z / temp
        z = mask_min_p(z, min_p)
        z = mask_top_k(z, top_k)
        z = mask_top_p(z, top_p)
        return softmax(z)
    if semantics == "llamacpp":
        z = mask_top_k(z, top_k)
        z = mask_top_p(z, top_p)
        z = mask_min_p(z, min_p)
        return softmax(z / temp)
    if semantics == "mlxlm":
        z = mask_top_p(z, top_p)
        z = mask_min_p(z, min_p)
        z = mask_top_k(z, top_k)
        return softmax(z / temp)
    raise ValueError(f"unknown semantics {semantics!r}")


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())
