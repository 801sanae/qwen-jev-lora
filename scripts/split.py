"""data/raw.jsonl -> data/train.jsonl, data/eval.jsonl (9:1, 레이블 층화)."""
import json, random
random.seed(42)
rows = [l for l in open("data/raw.jsonl", encoding="utf-8") if l.strip()]
by = {"Y": [], "N": []}
for l in rows:
    by[json.loads(l)["messages"][-1]["content"]].append(l)
tr, ev = [], []
for v in by.values():
    random.shuffle(v); k = max(1, len(v) // 10); ev += v[:k]; tr += v[k:]
random.shuffle(tr)
open("data/train.jsonl", "w", encoding="utf-8").writelines(tr)
open("data/eval.jsonl", "w", encoding="utf-8").writelines(ev)
print(f"train={len(tr)} eval={len(ev)}")
