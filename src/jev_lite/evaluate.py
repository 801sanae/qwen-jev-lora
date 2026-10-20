"""eval.jsonl 로 정확도·Brier·ECE 측정 (vLLM 서버 필요). 베이스/병합 모델을 같은 조건으로 비교할 것."""
import argparse, json
import httpx
from .scoring import p_yes

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/eval.jsonl")
p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
p.add_argument("--model", default="qwen-jev-custom")
a = p.parse_args()

with open(a.data, encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]
preds = []
with httpx.Client(timeout=30) as c:
    for r in rows:
        msgs, gold = r["messages"][:-1], r["messages"][-1]["content"]
        resp = c.post(a.url, json={"model": a.model, "messages": msgs, "max_tokens": 1,
                                   "temperature": 0, "logprobs": True, "top_logprobs": 5})
        resp.raise_for_status()
        body = resp.json()
        prob, cov = p_yes(body["choices"][0]["logprobs"]["content"][0]["top_logprobs"])
        preds.append((prob, 1 if gold == "Y" else 0, cov))

n = len(preds)
acc = sum((pr >= 0.5) == bool(y) for pr, y, _ in preds) / n
brier = sum((pr - y) ** 2 for pr, y, _ in preds) / n
ece = 0.0
for b in range(10):
    bucket = [(pr, y) for pr, y, _ in preds if b / 10 <= pr < (b + 1) / 10 or (b == 9 and pr == 1.0)]
    if bucket:
        ece += len(bucket) / n * abs(sum(p for p, _ in bucket) / len(bucket) - sum(y for _, y in bucket) / len(bucket))
print(json.dumps({"n": n, "accuracy": round(acc, 4), "brier": round(brier, 4), "ece": round(ece, 4),
                  "mean_coverage": round(sum(c for *_, c in preds) / n, 4)}, indent=2))
