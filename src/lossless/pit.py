"""Randomized probability integral transform (PIT) tests for token samplers.

If token y_t is drawn from q_t (conditionally on the past), then with tokens ordered by decreasing q_t,
    Z_t = F_t(y_t-) + U_t * q_t(y_t),   U_t ~ Unif(0,1) independent,
is exactly Unif(0,1), and Z_1, Z_2, ... are iid even though q_t depends adaptively on the past
(Rosenblatt 1952; Diebold, Gunther & Tay 1998; randomized for discrete outcomes: Brockwell 2007,
Czado, Gneiting & Held 2009). Coarsening the support (e.g. lumping every token outside a reported
top-K into one "tail" category) keeps the null exact; it only lowers power.

If the reference q_hat used for the transform differs from the distribution actually sampled from, the
law of Z differs from uniform by at most TV(q, q_hat): a noise floor that must be calibrated with a
negative control (speculation off).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


def pit_from_dist(probs: np.ndarray, token: int, u: float) -> float:
    """Z for one position given the full claimed distribution and the emitted token."""
    order = np.argsort(-probs, kind="stable")
    rank = int(np.nonzero(order == token)[0][0])
    f_minus = float(probs[order[:rank]].sum())
    return f_minus + u * float(probs[token])


def pit_from_topk(topk_probs: np.ndarray, hit_index: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Vectorized Z from per-position top-K probabilities (rows sorted descending).

    topk_probs: (N, K) claimed probabilities of the K most likely tokens at each position.
    hit_index:  (N,) index of the emitted token within the row, or -1 if it is outside the top-K
                (then it falls in the lumped tail category of mass 1 - sum(row)).
    u:          (N,) independent Unif(0,1).
    """
    topk_probs = np.asarray(topk_probs, dtype=np.float64)
    n, k = topk_probs.shape
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(topk_probs, axis=1)], axis=1)  # (N, K+1)
    head = cum[:, -1]
    tail = np.clip(1.0 - head, 0.0, 1.0)
    hi = np.asarray(hit_index)
    in_head = hi >= 0
    f_minus = np.where(in_head, cum[np.arange(n), np.clip(hi, 0, k - 1)], head)
    p_y = np.where(in_head, topk_probs[np.arange(n), np.clip(hi, 0, k - 1)], tail)
    return f_minus + u * p_y


def surprisal_terms(topk_probs: np.ndarray, hit_index: np.ndarray):
    """Per-position (surprisal, entropy, variance) of the coarsened variable (top-K + one tail bucket)."""
    p = np.asarray(topk_probs, dtype=np.float64)
    tail = np.clip(1.0 - p.sum(axis=1, keepdims=True), 0.0, 1.0)
    full = np.concatenate([p, tail], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(full > 0, np.log(full), 0.0)
    ent = -(full * logp).sum(axis=1)
    second = (full * logp**2).sum(axis=1)
    var = np.clip(second - ent**2, 0.0, None)
    k = p.shape[1]
    idx = np.where(np.asarray(hit_index) >= 0, hit_index, k)
    s = -logp[np.arange(p.shape[0]), idx]
    return s, ent, var


@dataclass
class TestReport:
    n: int
    ks_stat: float
    ks_p: float
    cvm_p: float
    chi2_p: float
    mean_z: float
    mean_z_p: float
    surprisal_z: float | None
    surprisal_p: float | None
    min_p: float

    def as_dict(self):
        return asdict(self)


def uniformity_tests(z: np.ndarray, bins: int = 20, surprisal=None) -> TestReport:
    """Battery of tests of Z ~ iid Unif(0,1). `surprisal` = (s, ent, var) arrays for the calibration z-test."""
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    ks = stats.kstest(z, "uniform")
    with np.errstate(all="ignore"):
        cvm = stats.cramervonmises(z, "uniform")
    cvm_p = float(cvm.pvalue)
    if not np.isfinite(cvm_p):  # asymptotic series breaks down for extreme statistics
        cvm_p = 0.0 if cvm.statistic > 1.0 else 1.0
    counts, _ = np.histogram(z, bins=bins, range=(0.0, 1.0))
    chi2 = stats.chisquare(counts)
    mz = float(z.mean())
    se = np.sqrt(1.0 / 12.0 / n)
    mz_p = float(2 * stats.norm.sf(abs(mz - 0.5) / se))
    sz = sp = None
    if surprisal is not None:
        s, ent, var = surprisal
        denom = np.sqrt(var.sum())
        if denom > 0:
            sz = float((s - ent).sum() / denom)
            sp = float(2 * stats.norm.sf(abs(sz)))
    ps = [float(ks.pvalue), cvm_p, float(chi2.pvalue), mz_p] + ([sp] if sp is not None else [])
    ps = [p if np.isfinite(p) else 0.0 for p in ps]
    return TestReport(n, float(ks.statistic), float(ks.pvalue), cvm_p, float(chi2.pvalue),
                      mz, mz_p, sz, sp, float(min(ps)))


def bonferroni_reject(report: TestReport, alpha: float = 0.01) -> bool:
    """Reject H0 if any test in the battery rejects at alpha / (number of tests)."""
    m = 5 if report.surprisal_p is not None else 4
    return report.min_p < alpha / m
