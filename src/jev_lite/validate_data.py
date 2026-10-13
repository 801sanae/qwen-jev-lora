"""train.jsonl 형식·레이블·클래스 균형 검증."""
import json, sys
from collections import Counter

REQUIRED = ("[STATE]", "[CRITERIA]", "[QUESTION]")


def main(path: str) -> None:
    counts, errors = Counter(), []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                msgs = json.loads(line)["messages"]
                user, asst = msgs[-2], msgs[-1]
                assert user["role"] == "user" and asst["role"] == "assistant"
                assert asst["content"] in ("Y", "N"), f"label={asst['content']!r}"
                assert all(t in user["content"] for t in REQUIRED), "template 누락"
                counts[asst["content"]] += 1
            except Exception as e:  # noqa: BLE001
                errors.append(f"line {i}: {e}")
    total = sum(counts.values())
    print(f"rows={total} Y={counts['Y']} N={counts['N']}")
    if total and abs(counts["Y"] - counts["N"]) / total > 0.1:
        print("WARN: 클래스 불균형 10% 초과")
    for e in errors:
        print("ERROR", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/train.jsonl")
