PY=python
sample:   ; $(PY) scripts/make_sample.py && $(PY) scripts/split.py
validate: ; $(PY) -m src.jev_lite.validate_data data/train.jsonl
train:    ; $(PY) -m src.jev_lite.train --train data/train.jsonl --eval data/eval.jsonl
merge:    ; $(PY) -m src.jev_lite.merge
serve:    ; ./scripts/serve_vllm.sh ./merged-model
gateway:  ; uvicorn src.jev_lite.server:app --host 127.0.0.1 --port 9000
evaluate: ; $(PY) -m src.jev_lite.evaluate --data data/eval.jsonl
