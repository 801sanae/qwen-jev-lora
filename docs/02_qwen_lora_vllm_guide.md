# Qwen2.5 소형 모델 LoRA 학습 및 vLLM 서빙 가이드

## 개요

Windows 데스크톱의 NVIDIA RTX 2060 계열 GPU(VRAM 8GB), RAM 32GB에서 WSL2를 이용해 Qwen2.5-0.5B-Instruct를 LoRA로 학습하고, 베이스 모델에 병합한 뒤 vLLM API로 서빙한다.

- 모델: `Qwen/Qwen2.5-0.5B-Instruct`
- 학습: Transformers + PEFT LoRA, FP16
- 배포: 병합 모델 + vLLM
- AirLLM 및 4-bit 양자화는 이번 실습에서 사용하지 않는다.
- 실제 장비에서 실행 검증한 코드는 아니다. vLLM 버전별 GPU backend 호환성을 설치 후 확인해야 한다.

## 1. WSL2 및 학습 환경 구성

vLLM은 Windows 네이티브를 지원하지 않으므로 WSL2 Ubuntu에서 실행한다. 학습과 서빙 환경은 분리한다.

Windows 관리자 PowerShell:

```powershell
wsl --install -d Ubuntu
```

설치 및 재부팅 후 Ubuntu 터미널:

```bash
nvidia-smi
sudo apt update
sudo apt install -y python3-venv
mkdir -p ~/qwen-ft
cd ~/qwen-ft
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install torch transformers peft accelerate datasets
```

GPU 확인:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

`CUDA available: True`인지 확인한다. 예시는 WSL 홈 디렉터리에 모델과 데이터를 저장한다.

## 2. 학습 데이터 작성

`train.jsonl`에 한 줄당 JSON 객체 하나를 넣는다. 아래 3개는 형식 확인용이다. 실제 실험에는 서로 다른 질문과 답변을 수십~수백 개 준비한다. 첫 실습 목적은 보험 지식 주입보다 답변 형식과 말투 조정으로 잡는다.

```jsonl
{"messages":[{"role":"user","content":"보험금 청구 서류는 어디에서 확인하나요?"},{"role":"assistant","content":"보험금 청구 서류는 보험사 공식 홈페이지 또는 앱의 보험금 청구 안내에서 확인할 수 있습니다. 상품과 청구 사유에 따라 필요한 서류가 다를 수 있습니다."}]}
{"messages":[{"role":"user","content":"계약을 해지하면 얼마를 받나요?"},{"role":"assistant","content":"해지환급금은 상품, 납입 기간, 계약 상태에 따라 달라집니다. 정확한 금액은 계약 조회 또는 고객센터를 통해 확인해 주세요."}]}
{"messages":[{"role":"user","content":"이 치료는 보험금이 무조건 나오나요?"},{"role":"assistant","content":"현재 정보만으로 지급 여부를 확정할 수 없습니다. 가입한 상품의 약관, 보장 내용, 진단 및 치료 내용을 확인해야 합니다."}]}
```

## 3. LoRA 학습

다음을 `train.py`로 저장한다. Qwen의 chat template을 사용하며 마지막 assistant 답변 토큰에만 loss를 적용한다.

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 384

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
model.config.use_cache = False

model = get_peft_model(
    model,
    LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    ),
)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="train.jsonl", split="train")


def encode(example):
    messages = example["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError("마지막 메시지는 assistant 답변이어야 합니다.")

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt와 full chat template의 prefix가 다릅니다.")

    input_ids = full_ids[:MAX_LENGTH]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


dataset = dataset.map(encode, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda row: any(x != -100 for x in row["labels"]))
if len(dataset) == 0:
    raise ValueError("학습 가능한 답변이 없습니다. 데이터 또는 MAX_LENGTH를 확인하세요.")


def collate(rows):
    length = max(len(row["input_ids"]) for row in rows)
    return {
        "input_ids": torch.tensor([
            row["input_ids"]
            + [tokenizer.pad_token_id] * (length - len(row["input_ids"]))
            for row in rows
        ]),
        "attention_mask": torch.tensor([
            row["attention_mask"]
            + [0] * (length - len(row["attention_mask"]))
            for row in rows
        ]),
        "labels": torch.tensor([
            row["labels"] + [-100] * (length - len(row["labels"]))
            for row in rows
        ]),
    }


trainer = Trainer(
    model=model,
    train_dataset=dataset,
    data_collator=collate,
    args=TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=42,
    ),
)

trainer.train()
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./adapter")
```

실행:

```bash
python train.py
```

전체 파라미터를 업데이트하는 Full FT가 아니라 `q_proj`, `v_proj`에 추가한 adapter만 학습한다. 짧은 입력과 batch size 1로 시작하며 VRAM 사용량은 실제 데이터로 확인한다.

## 4. 베이스 모델과 adapter 병합

첫 배포에서는 adapter 동적 로딩 대신 병합 모델 하나를 서빙한다. 다음을 `merge.py`로 저장한다.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
)

model = PeftModel.from_pretrained(base, "./adapter")
merged = model.merge_and_unload()
merged = merged.to(dtype=torch.float16)

merged.save_pretrained("./merged-model", safe_serialization=True)
AutoTokenizer.from_pretrained("./adapter").save_pretrained("./merged-model")
```

실행:

```bash
python merge.py
```

결과 구조:

```text
qwen-ft/
├─ train.jsonl
├─ train.py
├─ merge.py
├─ adapter/         # LoRA adapter
└─ merged-model/    # 병합 모델 및 tokenizer
```

## 5. vLLM 서빙 환경 및 실행

학습 프로세스 종료 후 별도 환경을 만든다.

```bash
deactivate
python3 -m venv .venv-serve
source .venv-serve/bin/activate
pip install --upgrade pip
pip install vllm
```

짧은 context, 동시 요청 1개, FP16 및 eager 실행으로 시작한다.

```bash
vllm serve ./merged-model \
  --served-model-name qwen-insurance-demo \
  --dtype half \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.65 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 8000
```

## 6. API 테스트

다른 WSL 터미널에서 실행한다.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-insurance-demo",
    "messages": [
      {
        "role": "user",
        "content": "보험금을 받을 수 있는지 지금 확정해 줄 수 있나요?"
      }
    ],
    "temperature": 0,
    "max_tokens": 160
  }'
```

## 7. 주의사항 및 성공 판정

- RTX 2060 계열에서는 `--dtype half`를 유지한다. 예시는 FP16 학습 및 서빙으로 구성했다.
- vLLM attention backend 지원 범위는 버전에 따라 다르다. FlashAttention 관련 오류가 발생하면 해당 버전의 Turing 지원 backend를 확인한다. 최신 문서에서는 `--attention-backend` 선택 옵션을 안내한다.
- VRAM 부족 시 브라우저와 Steam 등 불필요한 GPU 사용 앱을 종료한다.
- 학습과 서빙을 동시에 실행하지 않는다.
- 로컬 실습용 설정이다. 외부 공개 전에 인증과 방화벽을 구성한다.
- 금융권 실데이터 대신 합성 또는 비식별 데이터로 실습한다.
- 학습 loss 감소만으로 성공을 판단하지 않는다. 학습하지 않은 질문 10~20개로 베이스 모델과 병합 모델을 동일 조건에서 비교한다.
- 첫 단계는 0.5B로 파이프라인을 완성하는 것이다. 이후 모델 크기와 데이터 품질을 따로 개선한다.

## 문서 범위

이 문서는 앞선 대화에서 제공한 가이드를 Markdown으로 정리한 것이다. 패키지 버전은 고정하지 않았으며 설치 시점의 의존성과 실제 GPU 환경에 따라 조정이 필요할 수 있다.
