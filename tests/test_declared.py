"""declared.py against a brute-force engine with explicit tie rules (full vocabulary)."""
import numpy as np
import pytest

from lossless import declared as D
from lossless import pit as P


def engine_dist(z, temp, top_k, top_p, min_p, order, ties):
    """Brute-force engine: exact distribution over the full vocabulary. 'count' ties break by ascending id."""
    V = z.size
    keep = np.ones(V, bool)
    scale = 1.0
    ids = np.arange(V)
    for op in order:
        idx = np.lexsort((ids, -z))                       # descending value, then ascending id
        idx = idx[keep[idx]]
        if op == "temp":
            scale = 1.0 / temp
        elif op == "top_k" and top_k and top_k > 0 and top_k < idx.size:
            if ties["top_k"] == "count":
                new = np.zeros(V, bool); new[idx[:top_k]] = True
            else:
                new = keep & (z >= z[idx[top_k - 1]])
            keep = new
        elif op == "top_p" and top_p and 0 < top_p < 1:
            w = np.exp(z[idx] * scale); w /= w.sum()
            n = int(np.searchsorted(np.cumsum(w), top_p, side="left")) + 1
            n = min(n, idx.size)
            if ties["top_p"] == "count":
                new = np.zeros(V, bool); new[idx[:n]] = True
            else:
                new = keep & (z >= z[idx[n - 1]])
            keep = new
        elif op == "min_p" and min_p and min_p > 0:
            lw = z * scale
            thr = lw[keep].max() + np.log(min_p)
            keep = keep & (lw >= thr)
    q = np.where(keep, np.exp(z * scale - (z * scale)[keep].max()), 0.0)
    return q / q.sum()


def class_mass(q, z, values, merged, boundary):
    out = np.zeros(values.size - (1 if merged else 0) + (1 if merged else 0))
    for t in range(z.size):
        if q[t] == 0:
            continue
        hit = np.nonzero(values == z[t])[0]
        if hit.size and not (merged and hit[0] == values.size - 1):
            out[hit[0]] += q[t]
        elif merged:
            out[-1] += q[t]
        else:
            return None                                     # mass outside the declared categories
    return out


@pytest.mark.parametrize("semantics", ["hf", "vllm", "llamacpp", "mlxlm", "mlxlm_legacy"])
def test_declared_matches_bruteforce(semantics):
    rng = np.random.default_rng(1)
    order = D.ORDERS["mlxlm" if semantics == "mlxlm_legacy" else semantics]
    ties = D.TIES[semantics]
    n_ok = n_ok_topk = 0
    for trial in range(1500):
        V = int(rng.integers(8, 60))
        grid = rng.choice([0.25, 0.5, 1.0])
        logits = np.round(rng.normal(0, 2.5, V) / grid) * grid           # coarse grid: many ties
        z = logits - np.logaddexp.reduce(logits)
        temp = float(rng.choice([0.6, 0.7, 1.0, 1.3]))
        top_k = int(rng.choice([0, 1, 3, 5, 10, 20, 40]))
        top_p = float(rng.choice([1.0, 0.95, 0.9, 0.8, 0.5]))
        min_p = float(rng.choice([0.0, 0.0, 0.05, 0.1]))
        q = engine_dist(z, temp, top_k, top_p, min_p, order, ties)
        sem = "mlxlm" if semantics == "mlxlm_legacy" else semantics
        d = D.declare(z, temp, top_k, top_p, min_p, sem, ties=ties, has_tail=False)
        if d.status == "ok":
            cm = class_mass(q, z, d.values, d.merged_tail, d.boundary_value)
            assert cm is not None and np.allclose(cm, d.mass, atol=1e-9), (semantics, trial)
            n_ok += 1
        # top-K report + tail mass
        K = int(rng.integers(3, 25))
        top = np.argsort(-z, kind="stable")[:K]
        d2 = D.declare(z[top], temp, top_k, top_p, min_p, sem, ties=ties, has_tail=True)
        if d2.status == "ok":
            cm = class_mass(q, z, d2.values, d2.merged_tail, d2.boundary_value)
            assert cm is not None and np.allclose(cm, d2.mass, atol=1e-9), (semantics, trial, "topK")
            n_ok_topk += 1
    assert n_ok > 1000 and n_ok_topk > 300, (n_ok, n_ok_topk)


def test_declared_pit_uniform_under_null():
    """Engine samples from its exact distribution; the PIT over declared categories from a top-20 report is uniform."""
    rng = np.random.default_rng(7)
    zs = []
    for _ in range(20000):
        V = 50
        logits = np.round(rng.normal(0, 2.0, V) / 0.5) * 0.5
        z = logits - np.logaddexp.reduce(logits)
        q = engine_dist(z, 0.7, 20, 0.9, 0.0, D.ORDERS["llamacpp"], D.TIES["llamacpp"])
        y = rng.choice(V, p=q)
        top = np.argsort(-z, kind="stable")[:20]
        d = D.declare(z[top], 0.7, 20, 0.9, 0.0, "llamacpp", has_tail=True)
        if d.status != "ok":
            continue
        zz, *_ , viol = D.pit(d, float(z[y]), bool(y in set(top.tolist())), rng.random())
        assert not viol
        zs.append(zz)
    rep = P.uniformity_tests(np.array(zs))
    assert len(zs) > 15000 and rep.ks_p > 1e-3 and not P.bonferroni_reject(rep, 0.01)


def test_zero_tail_report_is_complete():
    """A top-K report whose probabilities sum to 1 (confident position) leaves nothing unreported: the boundary
    class is complete and the declared distribution must be computable at any temperature."""
    z = np.log(np.array([0.9, 0.06, 0.04]))
    for T in (1.0, 0.6):
        d = D.declare(z, T, 0, 1.0, 0.0, "llamacpp", has_tail=True)
        assert d.status == "ok" and not d.merged_tail and np.isclose(d.mass.sum(), 1.0), (T, d.status, d.reason)
