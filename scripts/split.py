"""data/raw.jsonl -> data/train.jsonl, data/eval.jsonl (9:1, 레이블 층화)."""
import json, random
random.seed(42)
with open("data/raw.jsonl", encoding="utf-8") as f:
    rows = [l for l in f if l.strip()]
by = {"Y": [], "N": []}
for l in rows:
    by[json.loads(l)["messages"][-1]["content"]].append(l)
tr, ev = [], []
for v in by.values():
    random.shuffle(v); k = max(1, len(v) // 10); ev += v[:k]; tr += v[k:]
random.shuffle(tr)
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    f.writelines(tr)
with open("data/eval.jsonl", "w", encoding="utf-8") as f:
    f.writelines(ev)
print(f"train={len(tr)} eval={len(ev)}")
