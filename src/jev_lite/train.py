"""LoRA 학습 (RTX 2060 8GB / FP16 기준). 원본 가이드 대비 변경점:
- LoRA 파라미터를 FP32로 유지 (FP16 가중치 + fp16 AMP 시 'Attempting to unscale FP16 gradients' 오류 방지)
- target_modules 확장(q,k,v,o), eval 셋 지원, CLI 인자화
"""
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .preflight import require_cuda

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--train", default="data/train.jsonl")
p.add_argument("--eval", default=None)
p.add_argument("--out", default="./adapter")
p.add_argument("--max-length", type=int, default=768)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--rank", type=int, default=16)
args = p.parse_args()
use_cuda = require_cuda("LoRA 학습")  # CPU 진행 시 FP16 AMP 불가 -> FP32

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16 if use_cuda else torch.float32, attn_implementation="eager")
model.config.use_cache = False
model.enable_input_require_grads()  # gradient checkpointing + PEFT
model = get_peft_model(model, LoraConfig(
    task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
for prm in model.parameters():
    if prm.requires_grad:
        prm.data = prm.data.float()
model.print_trainable_parameters()


def encode(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False)
    assert full[:len(prompt)] == prompt, "chat template prefix 불일치"
    if len(full) > args.max_length:  # 앞쪽 STATE를 자르면 정답 근거가 사라지므로 행 자체를 버림
        return {"input_ids": [], "attention_mask": [], "labels": []}
    labels = [-100] * len(prompt) + full[len(prompt):]  # 'Y'/'N' + <|im_end|>
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


def load(path):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(encode, remove_columns=ds.column_names)
    return ds.filter(lambda r: len(r["input_ids"]) > 0)


train_ds = load(args.train)
eval_ds = load(args.eval) if args.eval else None
print(f"train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")


def collate(rows):
    n = max(len(r["input_ids"]) for r in rows)
    pad = lambda r, k, v: r[k] + [v] * (n - len(r[k]))  # noqa: E731
    return {
        "input_ids": torch.tensor([pad(r, "input_ids", tok.pad_token_id) for r in rows]),
        "attention_mask": torch.tensor([pad(r, "attention_mask", 0) for r in rows]),
        "labels": torch.tensor([pad(r, "labels", -100) for r in rows]),
    }


Trainer(
    model=model, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collate,
    args=TrainingArguments(
        output_dir="./checkpoints", per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_steps=0.05, lr_scheduler_type="cosine",
        fp16=use_cuda, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, eval_strategy="epoch" if eval_ds else "no", save_strategy="no",
        report_to="none", seed=42),
).train()
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
