# qwen-jev-lora

Qwen2.5-0.5B-Instruct를 LoRA로 학습해 **Jev 스타일 System-1 판정 모델**(Y/N 1토큰 + 보정 확률)을 만들고, vLLM + `POST /v1/systemone` 호환 게이트웨이로 서빙하는 학습용 레포입니다.

## 구조
```
docs/        원본 분석·가이드 문서
prompts/     상위 LLM용 합성 데이터 생성 프롬프트
data/        raw/train/eval jsonl (train·eval 은 gitignore)
scripts/     샘플 생성, split, vLLM 실행
pyproject.toml / uv.lock  의존성 그룹 train·serve
src/jev_lite/      (uv_build 패키지, `import jev_lite`)
  template.py      [domain] STATE/CRITERIA/QUESTION 템플릿 (학습·추론 공용)
  scoring.py       [domain] top_logprobs -> P(Y)
  validate_data.py [app]    데이터 형식·균형 검증
  train.py / merge.py [infra] LoRA 학습·병합
  server.py        [interface] /v1/systemone -> vLLM
  evaluate.py      정확도·Brier·ECE
  preflight.py     CUDA 점검·로깅, 불가 시 진행 여부 확인
```

## 빠른 시작 (WSL2 Ubuntu, RTX 2060 8GB)
[uv](https://docs.astral.sh/uv/) 하나로 환경을 관리합니다 (`pyproject.toml` + `uv.lock`, Python 3.12).
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group train                     # serve(기본) + train 의존성을 .venv 에 설치
make sample validate train merge          # 형식 확인용 샘플
make serve                                # 터미널 1 : vLLM :8000
make gateway                              # 터미널 2 : jev-lite :9000
make evaluate                             # 터미널 3
```
- `train` 그룹(torch)은 GPU 장비에서만 설치하세요. macOS 에서는 `uv sync` 로 게이트웨이·평가·데이터 스크립트만 설치되고 vLLM 은 제외됩니다(Intel Mac 은 torch 휠이 없음).
- 직접 실행: `uv run -m jev_lite.train --epochs 5`, `uv run uvicorn jev_lite.server:app --port 9000`
- 학습과 서빙을 동시에 돌리지 마세요(VRAM).
- `make train` / `make serve` 는 시작 전에 CUDA를 점검하고 GPU 정보를 로깅합니다(`make check` 로 단독 실행 가능). CUDA가 없으면 계속할지 묻고, 비대화형 환경에서는 중단합니다(강제 진행: `JEV_ALLOW_NO_CUDA=1`). CPU로 학습하면 FP32로 전환됩니다.

호출 예:
```bash
curl -s localhost:9000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"contract_days": 120, "inpatient_days": 4},
  "questions": {"is_eligible": {"instructions": "입원일당 지급 대상인가?",
    "criteria": {"true": "입원 3일 이상, 면책 90일 경과", "false": "그 외"}}}}'
```

## 실제 데이터 만들기
1. `prompts/data_generation_prompt.md`의 도메인·개수를 채워 상위 LLM에 요청 (도메인당 최소 수백 행 권장)
2. 결과를 `data/raw.jsonl`에 합치고 `uv run scripts/split.py` → `make validate`
3. **학습 템플릿과 서빙 템플릿(`template.py`)이 1글자라도 다르면 성능이 무너집니다.** 생성 데이터도 같은 포맷인지 검증하세요.

## 원본 가이드 대비 수정 사항
| 항목 | 원본 | 이 레포 | 이유 |
|---|---|---|---|
| LoRA 파라미터 dtype | FP16 | FP32 | FP16 가중치 + `fp16=True` 조합은 GradScaler `unscale FP16 gradients` 오류 |
| target_modules | q,v | q,k,v,o | 분류형 과제에서 표현력 확보 (VRAM 여유 충분) |
| 긴 샘플 | 뒤를 truncate | 행 제거 | 정답 토큰이 잘리면 학습 불가 |
| P(Y) | "Jev noul과 동일" | Y/N 재정규화 + coverage | 아래 주의 참고 |
| 평가 | 수동 비교 | acc / Brier / ECE | 확률 출력 모델은 보정(calibration)이 핵심 |

## 주의 (꼭 읽기)
- **확률 ≠ 보정된 확률**: 1토큰 softmax는 과신(overconfident) 경향이 큽니다. Jev의 calibrated probability를 흉내 내려면 eval 셋으로 temperature scaling / Platt scaling을 추가하세요. `coverage`(Y+N 확률합)가 낮으면 형식이 무너진 신호입니다.
- **합성 데이터 편향**: 단일 LLM이 만든 데이터는 표현이 단조롭고 레이블 오류가 섞입니다. 소량이라도 사람 검수 골든셋을 따로 두세요.
- **역공학 관련**: `docs/01_jev_analysis.md`는 공개 SDK의 인터페이스 형태만 참고합니다. 상용 SDK 라이선스(역컴파일 금지 조항 등)를 확인하고, 내부 사용 시 원 제품 명칭·코드를 포함하지 마세요.
- **금융권**: 실고객 데이터 금지(합성·비식별만). 게이트웨이는 인증이 없으므로 127.0.0.1 바인딩 유지. 판정 결과는 보조 지표로만 쓰고 최종 심사 로직에 단독 사용하지 마세요.
- 이 코드는 실제 GPU 장비에서 검증되지 않았습니다. vLLM 버전에 따라 Turing(sm75) attention backend 옵션 확인이 필요합니다.

## 다음 단계
- `choice`(다자택일: 선택지별 첫 토큰 logprob), `score`(0~9 토큰 기대값) 타입 추가
- 다중 질문 시 prefix caching(`--enable-prefix-caching`)으로 STATE 재사용
- 1.5B/3B 모델 비교, temperature scaling 모듈 추가
