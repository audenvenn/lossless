"""Minimal reproduction of a silent top-k cap: count distinct FIRST tokens over many seeds at T=1, top_k=0 (no
truncation requested) on a high-entropy prompt. An engine that caps top-k at K can never produce more than K
distinct first tokens. Usage: python -I first_token_support.py URL MODEL N OUT.json
"""
import json
import sys
import urllib.request
from collections import Counter

PROMPT = "Write one random English word. Reply with the word only."


def main():
    url, model, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    firsts = Counter()
    for seed in range(n):
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "temperature": 1.0, "top_p": 1.0,
                "top_k": 0, "min_p": 0.0, "max_tokens": 1, "seed": seed, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        m = r["choices"][0]["message"]
        firsts[(m.get("content") or "") + "|" + (m.get("reasoning_content") or "")] += 1
    res = {"n": n, "distinct_first_tokens": len(firsts), "top": firsts.most_common(10)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"n": n, "distinct_first_tokens": len(firsts)}))


if __name__ == "__main__":
    main()
