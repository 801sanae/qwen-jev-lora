import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--adapter", default="./adapter")
p.add_argument("--out", default="./merged-model")
a = p.parse_args()

base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, device_map="cpu")
merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload().to(torch.float16)
merged.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
print("saved:", a.out)
