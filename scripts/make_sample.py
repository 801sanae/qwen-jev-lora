"""형식 확인용 합성 샘플 생성 (실데이터 아님). 실제 학습은 prompts/ 로 상위 LLM에서 수백~수천 행 생성."""
import json, sys
sys.path.insert(0, "src")
from jev_lite.template import render_prompt

C = {"true": "입원일수가 3일 이상이고 계약 후 90일 면책기간이 지난 경우",
     "false": "입원일수 3일 미만이거나 면책기간 내 발생, 또는 정보 불충분"}
Q = "이 청구 건은 입원일당 지급 대상인가?"
cases = [
    ({"contract_days": 400, "inpatient_days": 5}, "Y"),
    ({"contract_days": 400, "inpatient_days": 2}, "N"),
    ({"contract_days": 91, "inpatient_days": 3}, "Y"),   # 경계
    ({"contract_days": 89, "inpatient_days": 10}, "N"),  # 경계
    ("가입한 지 2년 됐고 폐렴으로 4박 5일 입원했어요.", "Y"),
    ("지난달 가입했는데 맹장수술로 일주일 입원했습니다.", "N"),
    ("입원은 했는데 며칠인지는 기억 안 나요. 가입은 오래됐어요.", "N"),
    ({"contract_days": 1200, "inpatient_days": 3}, "Y"),
]
with open("data/raw.jsonl", "w", encoding="utf-8") as f:
    for s, y in cases * 5:
        f.write(json.dumps({"messages": [{"role": "user", "content": render_prompt(s, Q, C)},
                                         {"role": "assistant", "content": y}]}, ensure_ascii=False) + "\n")
print("wrote data/raw.jsonl")
