import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lossless import pit, samplers, toy  # noqa: E402


def test_transform_semantics_agree_without_truncation():
    rng = np.random.default_rng(1)
    z = rng.normal(size=200) * 3
    ref = samplers.transform(z, temp=0.7, semantics="hf")
    for s in ("vllm", "llamacpp", "mlxlm"):
        assert np.allclose(samplers.transform(z, temp=0.7, semantics=s), ref, atol=1e-14)


def test_transform_semantics_differ_with_truncation():
    rng = np.random.default_rng(2)
    z = rng.gumbel(size=500) * 2
    a = samplers.transform(z, temp=0.6, top_p=0.95, top_k=20, semantics="hf")
    b = samplers.transform(z, temp=0.6, top_p=0.95, top_k=20, semantics="llamacpp")
    assert samplers.tv(a, b) > 0  # different supports in general
    assert abs(a.sum() - 1) < 1e-12 and abs(b.sum() - 1) < 1e-12


def test_pit_exact_uniform_under_null_full_and_topk():
    rng = np.random.default_rng(3)
    n, V, K = 40000, 300, 10
    logits = rng.gumbel(size=(n, V)) * rng.uniform(0.5, 6, size=(n, 1))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    y = (cdf < rng.random((n, 1))).sum(axis=1)
    # top-K coarsening (tail lumped)
    order = np.argsort(-probs, axis=1, kind="stable")
    topk_idx = order[:, :K]
    topk_p = np.take_along_axis(probs, topk_idx, axis=1)
    hit = np.full(n, -1)
    match = topk_idx == y[:, None]
    has = match.any(axis=1)
    hit[has] = match[has].argmax(axis=1)
    z = pit.pit_from_topk(topk_p, hit, rng.random(n))
    assert stats.kstest(z, "uniform").pvalue > 1e-3
    rep = pit.uniformity_tests(z, surprisal=pit.surprisal_terms(topk_p, hit))
    assert not pit.bonferroni_reject(rep, alpha=1e-3)


def test_reference_speculative_sampler_is_lossless():
    rng = np.random.default_rng(4)
    m = toy.ToyLM.random(V=32, seed=5)
    P, Q = m.tables(temp=0.7, top_k=10, top_p=0.95)
    P1, Q1 = m.tables(temp=1.0)
    c1, c2, y = toy.simulate_spec(m, P, Q, P1, Q1, K=4, n_tokens=200000, variant="correct", rng=rng)
    z = toy.pit_values(P, c1, c2, y, rng)
    rep = pit.uniformity_tests(z, surprisal=toy.surprisal_from_tables(P, c1, c2, y))
    assert not pit.bonferroni_reject(rep, alpha=1e-3), rep


def test_argmax_on_reject_is_detected():
    rng = np.random.default_rng(6)
    m = toy.ToyLM.random(V=32, seed=7)
    P, Q = m.tables(temp=0.8)
    P1, Q1 = m.tables(temp=1.0)
    c1, c2, y = toy.simulate_spec(m, P, Q, P1, Q1, K=4, n_tokens=100000, variant="argmax_on_reject", rng=rng)
    z = toy.pit_values(P, c1, c2, y, rng)
    rep = pit.uniformity_tests(z, surprisal=toy.surprisal_from_tables(P, c1, c2, y))
    assert pit.bonferroni_reject(rep, alpha=1e-3), rep


def test_twosample_permutation_null_and_alternative():
    from lossless import twosample
    rng = np.random.default_rng(8)
    prompts = np.repeat(np.arange(20), 4)
    arm = np.tile([0, 1, 0, 1], 20)
    null_vals = [rng.random(200) for _ in prompts]
    r0 = twosample.permutation_test(prompts, arm, null_vals, n_perm=500)
    assert min(r0["p"].values()) > 0.001
    alt_vals = [rng.random(200) ** (1.15 if a == 1 else 1.0) for a in arm]  # arm 1 skewed toward 0
    r1 = twosample.permutation_test(prompts, arm, alt_vals, n_perm=500)
    assert r1["p"]["mean"] < 0.01


def _write_openai_jsonl(path, n_seq, length, temp_true, top_k, rng, K=20, V=200):
    """Synthetic engine output: tokens sampled at temp_true from random logits, reported as OpenAI-style
    top_logprobs of the *untransformed* distribution (what most servers report)."""
    import json as _json
    with open(path, "w") as f:
        for _ in range(n_seq):
            content = []
            for _ in range(length):
                z = rng.gumbel(size=V) * rng.uniform(0.5, 4)
                lp = z - np.log(np.exp(z - z.max()).sum()) - z.max()
                q = samplers.transform(z, temp_true, top_k, 1.0, 0.0, "hf")
                y = int(rng.choice(V, p=q))
                order = np.argsort(-lp)[:K]
                tops = [{"token": f"t{int(i)}", "logprob": float(lp[i])} for i in order]
                content.append({"token": f"t{y}", "logprob": float(lp[y]), "top_logprobs": tops})
            f.write(_json.dumps({"logprobs": {"content": content}}) + "\n")


def test_cli_l1_jsonl_openai_logprobs(tmp_path):
    from lossless import cli
    rng = np.random.default_rng(11)
    p = tmp_path / "ok.jsonl"
    # HF keeps every token tied with the k-th value, so the report must reach past rank top_k (K = 25 here)
    _write_openai_jsonl(p, n_seq=40, length=200, temp_true=0.7, top_k=20, rng=rng, K=25)
    ok = cli._l1_from_jsonl(str(p), temp=0.7, top_p=1.0, top_k=20, min_p=0.0, semantics="hf")
    assert ok["tested"] == ok["tokens"] and not ok["reject_bonf_0.01"] and ok["support_violations"] == 0
    bad = cli._l1_from_jsonl(str(p), temp=0.9, top_p=1.0, top_k=20, min_p=0.0, semantics="hf")
    assert bad["reject_bonf_0.01"]
    # a top-20 report with top_k = 20 cannot rule out unreported ties under HF's rule: positions are screened
    p2 = tmp_path / "k20.jsonl"
    _write_openai_jsonl(p2, n_seq=4, length=50, temp_true=0.7, top_k=20, rng=rng, K=20)
    r2 = cli._l1_from_jsonl(str(p2), temp=0.7, top_p=1.0, top_k=20, min_p=0.0, semantics="hf")
    assert r2["tested"] == 0
    # ... but it can under llama.cpp's 'count' rule (sort + resize keeps exactly k)
    r3 = cli._l1_from_jsonl(str(p2), temp=0.7, top_p=1.0, top_k=20, min_p=0.0, semantics="llamacpp")
    assert r3["tested"] == r3["tokens"]
