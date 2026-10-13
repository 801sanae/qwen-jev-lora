"""Interface: Jev 호환 POST /v1/systemone  ->  vLLM chat.completions(max_tokens=1, logprobs).
실행: uvicorn src.jev_lite.server:app --port 9000
"""
import asyncio, os
from typing import Any, Literal
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from .template import render_prompt
from .scoring import p_yes

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.getenv("SERVED_MODEL", "qwen-jev-custom")
app = FastAPI(title="jev-lite")


class Criteria(BaseModel):
    true: str | None = None
    false: str | None = None


class Question(BaseModel):
    type: Literal["noul"] = "noul"  # choice/score 는 TODO (docs 참고)
    instructions: str
    criteria: Criteria | None = None


class SystemOneRequest(BaseModel):
    state: Any = None
    questions: dict[str, Question]


async def ask(client: httpx.AsyncClient, state: Any, q: Question) -> tuple[dict, dict]:
    prompt = render_prompt(state, q.instructions, q.criteria.model_dump() if q.criteria else None)
    r = await client.post(VLLM_URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5})
    r.raise_for_status()
    body = r.json()
    top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    prob, cov = p_yes(top)
    return {"type": "noul", "noul": round(prob, 4), "coverage": round(cov, 4)}, body["usage"]


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest):
    async with httpx.AsyncClient(timeout=30) as c:
        keys = list(req.questions)
        results = await asyncio.gather(*(ask(c, req.state, req.questions[k]) for k in keys))
    return {
        "model": "jev-lite",
        "answers": {k: a for k, (a, _) in zip(keys, results)},
        "usage": {"input_tokens": sum(u["prompt_tokens"] for _, u in results),
                  "output_tokens": sum(u["completion_tokens"] for _, u in results)},
    }
