"""Domain: Jev 스타일 프롬프트 템플릿 (학습/추론 공용 - 반드시 동일해야 함)."""
import json
from typing import Any

LABELS = ("Y", "N")


def render_prompt(state: Any, instructions: str, criteria: dict | None = None) -> str:
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    criteria = criteria or {}
    lines = ["[STATE]", state or "null", "", "[CRITERIA]"]
    lines.append(f"- True (Y): {criteria.get('true', '질문 명제가 참이면 Y')}")
    lines.append(f"- False (N): {criteria.get('false', '그 외에는 N')}")
    lines += ["", "[QUESTION]", f"{instructions.strip()} (Y/N)"]
    return "\n".join(lines)
