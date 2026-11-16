"""L3 (black-box) generation client for any OpenAI-compatible engine server.

Sends identical chat requests (same prompts, same sampling parameters) for one arm (e.g. speculation on or off,
selected by how the server was launched) and saves the raw replies for a fixed-reference scoring pass
(l3_score.py). Requests per-request seeds and top logprobs when the server accepts them (ignored otherwise).

Usage: python -I l3_client.py --base-url http://127.0.0.1:PORT --model NAME --label kapitsa --arm on \
          --setting qwen_think_T0.6_p0.95_k20 --n-prompts 32 --samples 2 --max-tokens 256 --out results/l3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

SETTINGS = {
    "T1.0_plain": (1.0, 1.0, 0.0, 0, False),
    "qwen_nothink_T0.7_p0.8_k20": (0.7, 0.8, 0.0, 20, False),
    "qwen_think_T0.6_p0.95_k20": (0.6, 0.95, 0.0, 20, True),
}


def post(url: str, body: dict, timeout: int = 900) -> dict:
    """POST with retries on 503 (server still loading) and connection errors, up to ~2 minutes."""
    delay = 1.0
    for attempt in range(12):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 503 or attempt == 11:
                raise
        except urllib.error.URLError:
            if attempt == 11:
                raise
        time.sleep(delay); delay = min(delay * 2, 15.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--logprobs", type=int, default=20, help="top_logprobs to request (0 = don't ask)")
    ap.add_argument("--extra-body", default="", help="JSON object merged into every request body")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "l3")
    a = ap.parse_args()
    temp, top_p, min_p, top_k, think = SETTINGS[a.setting]
    prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text())[: a.n_prompts]
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.label}_{a.setting}_arm{a.arm}.jsonl"
    url = a.base_url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    n = 0
    ask_lp = a.logprobs > 0
    with path.open("w") as f:
        for pi, ptxt in enumerate(prompts):
            for s in range(a.samples):
                body = {"model": a.model, "messages": [{"role": "user", "content": ptxt}],
                        "temperature": temp, "top_p": top_p, "top_k": top_k, "min_p": min_p,
                        "max_tokens": a.max_tokens, "stream": False,
                        "seed": a.seed + 1000 * pi + s,
                        "chat_template_kwargs": {"enable_thinking": think}}
                if ask_lp:
                    body |= {"logprobs": True, "top_logprobs": a.logprobs}
                if a.extra_body:
                    body |= json.loads(a.extra_body)
                ts = time.time()
                try:
                    r = post(url, body)
                except urllib.error.HTTPError as e:
                    msg = e.read().decode(errors="replace")[:300]
                    if ask_lp and ("logprob" in msg.lower() or e.code in (400, 422)):
                        ask_lp = False  # retry without logprobs, for the whole run
                        body.pop("logprobs", None); body.pop("top_logprobs", None)
                        r = post(url, body)
                    else:
                        raise
                dt = time.time() - ts
                ch = r["choices"][0]
                msg = ch.get("message", {})
                rec = {"prompt": pi, "sample": s, "seed": body["seed"], "elapsed": dt,
                       "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content")
                       or msg.get("reasoning") or "", "finish_reason": ch.get("finish_reason"),
                       "usage": r.get("usage"), "logprobs": ch.get("logprobs"), "asked_logprobs": ask_lp}
                f.write(json.dumps(rec) + "\n"); f.flush()
                n += 1
            if pi % 8 == 7:
                print(f"[{a.label}/{a.arm}] {pi+1}/{len(prompts)} prompts, {n} replies, {time.time()-t0:.0f}s",
                      flush=True)
    print(f"wrote {path} ({n} replies, {time.time()-t0:.0f}s, logprobs={'yes' if ask_lp else 'no'})")


if __name__ == "__main__":
    main()
