# Jev 역공학 분석 및 LoRA 학습 데이터 생성 프롬프트 가이드

TypeSafe AI의 Jev SDK(`@typesafe-ai/sdk` v0.6.0)를 REA로 정적/구조 분석한 결과를 바탕으로, **Qwen2.5-0.5B-Instruct 모델을 유사 Jev(초고속 의사결정 모델)로 LoRA 파인튜닝하기 위한 학습 데이터 생성 프롬프트**를 정의합니다.

---

## 1. Jev 핵심 아키텍처 역공학 결과 요약

### 1.1 통신 및 엔드포인트 규격
* **엔드포인트**: `POST /v1/systemone`
* **기본 모델 식별자**: `jev-latest`
* **모델 특성**: 토큰 단위 문장 생성을 하지 않는 **System One(비정형 생성 배제형) 의사결정 모델**

### 1.2 입력 페이로드 (`SystemOneRequest`)
```json
{
  "state": "...",           // 평가 대상 데이터 (텍스트, JSON, Array, null)
  "questions": {
    "is_eligible": {
      "type": "noul",       // noul (Yes/No), choice (다자택일), score (점수 루브릭)
      "instructions": "질문 본문",
      "criteria": {
        "true": "Yes 판정 기준 설명 (optional)",
        "false": "No 판정 기준 설명 (optional)"
      }
    }
  }
}
```

### 1.3 출력 페이로드 (`SystemOneResult`)
```json
{
  "model": "jev-latest",
  "answers": {
    "is_eligible": {
      "type": "noul",
      "noul": 0.9412        // 0.0 ~ 1.0 사이의 정밀한 Yes 확률(Calibrated Probability)
    }
  },
  "usage": { "input_tokens": 128, "output_tokens": 1 }
}
```

> **핵심 원리**: Jev의 출력 토큰은 실질적으로 1개(`output_tokens: 1`)이며, 내부의 확률 분포(Logits)를 기반으로 신뢰도(Confidence/Probability)를 계산합니다.

---

## 2. Qwen2.5-0.5B LoRA 훈련 데이터 포맷 설계

vLLM에서 `max_tokens=1` 및 `logprobs=true`로 초고속 서빙하기 위한 Chat Template 표준 규격입니다.

### 2.1 Prompt 템플릿 (User 입력)
```text
[STATE]
{평가할 상태, 고객 문의, 서류 내용, 데이터 JSON 등}

[CRITERIA]
- True (Y): {Yes로 판정할 구체적 기준}
- False (N): {No로 판정할 구체적 기준}

[QUESTION]
{판정 질문} (Y/N)
```

### 2.2 Target 레이블 (Assistant 응답)
* **`Y`** 또는 **`N`** (단일 토큰)
* *추론 시 vLLM이 반환하는 Y 토큰의 logprob를 Softmax 취하면 Jev의 `noul` 값(0.0~1.0)과 동일해집니다.*

---

## 3. 상위 LLM에 요청할 학습 데이터 생성 프롬프트

아래 프롬프트 전체를 복사하여 Claude, GPT-4o 등의 상위 LLM에 전달하면, 파인튜닝에 바로 사용할 수 있는 고품질 `train.jsonl` 데이터를 자동으로 생성받을 수 있습니다.

```markdown
당신은 LLM PEFT(LoRA) 파인튜닝용 합성 데이터셋을 구축하는 고성능 데이터 엔지니어입니다.
목표는 경량 LLM(Qwen2.5-0.5B)을 TypeSafe AI의 "Jev"와 같은 고속 의사결정/불리언 판정(System 1 Decision) 모델로 만들기 위한 훈련 데이터셋(train.jsonl)을 생성하는 것입니다.

### 1. 도메인 및 작업 목표
- 도메인: [여기에 원하는 도메인 입력, 예: 보험금 청구 심사 / 고객센터 라우팅 / 금융 이상거래 탐지 등]
- 모델 역할: 입력된 상태(State)와 기준(Criteria)을 엄격히 대조하여 질문(Question)에 대해 오직 'Y' 또는 'N' 단 1글자만 출력

### 2. 데이터 형식 요구사항
각 데이터 행은 유효한 JSON 문자열이어야 하며, OpenAI Chat 포맷을 따릅니다:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "Y 또는 N"}]}

User 프롬프트 구조는 반드시 아래 템플릿을 엄격하게 준수하세요:
[STATE]
<상황, 고객 요청 본문, 문서 내용, 트랜잭션 데이터 등 구체적인 사실 정보>

[CRITERIA]
- True (Y): <Y로 판정하기 위한 명확하고 구체적인 규칙/조건>
- False (N): <N으로 판정하기 위한 조건 또는 예외 규정>

[QUESTION]
<Y 또는 N으로 답이 떨어지는 명확한 판정 명제> (Y/N)

### 3. 데이터셋 생성 규칙 (중요)
1. **정답 레이블의 엄격성**: Assistant의 `content`는 오직 "Y" 또는 "N" 단 한 글자여야 합니다. 불필요한 공백, 마침표, 부가 설명("네", "아니오", "Y입니다" 등)을 절대 포함하지 마세요.
2. **5:5 클래스 균형**: Y와 N의 정답 비율을 정확히 5:5 비율로 균등하게 배분하세요.
3. **경계 케이스(Edge Case) 포함**: 명백한 케이스 외에도 기준에 아슬아슬하게 부합하거나 부합하지 않는 애매한 상황을 20% 이상 포함하세요.
4. **다양성**: [STATE]의 표현 방식(구어체, 정형 텍스트, JSON 등)과 [CRITERIA]의 규칙 복잡도를 다양하게 변형하세요.

### 4. 출력 포맷
- 다른 설명이나 머리말/꼬리말 없이 오직 순수한 JSONL 코드 블록(```jsonl ... ```)만 출력하세요.
- 총 [원하는 개수, 예: 50]개의 행을 생성해 주세요.
```

---

## 4. 학습 및 배포 파이프라인 연계 요약

1. **데이터 수집**: 상위 LLM에서 위 프롬프트로 `train.jsonl` 획득
2. **LoRA 파인튜닝**: `train.py` 실행 (RTX 2060 8GB VRAM에서 약 5~10분 소요)
3. **가중치 병합**: `merge.py` 실행으로 FP16 단일 모델 생성
4. **vLLM 서빙**:
   ```bash
   vllm serve ./merged-model --max-model-len 1024 --dtype half --enforce-eager
   ```
5. **Jev 스타일 호출**:
   ```bash
   curl http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "qwen-jev-custom",
       "messages": [{"role": "user", "content": "[STATE]\n..."}],
       "max_tokens": 1,
       "logprobs": true,
       "top_logprobs": 2
     }'
   ```
   * 반환된 logprobs에서 `Y`의 확률을 산출하면 Jev의 `noul` 값과 동일한 결과를 획득합니다.
