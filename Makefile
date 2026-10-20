RUN=uv run
TRAIN=uv run --group train
check:    ; $(RUN) -m jev_lite.preflight "환경 점검"
sample:   ; $(RUN) scripts/make_sample.py && $(RUN) scripts/split.py
validate: ; $(RUN) -m jev_lite.validate_data data/train.jsonl
train:    ; $(TRAIN) -m jev_lite.train --train data/train.jsonl --eval data/eval.jsonl
merge:    ; $(TRAIN) -m jev_lite.merge
serve:    ; $(RUN) ./scripts/serve_vllm.sh ./merged-model
gateway:  ; $(RUN) uvicorn jev_lite.server:app --host 127.0.0.1 --port 9000
evaluate: ; $(RUN) -m jev_lite.evaluate --data data/eval.jsonl
.PHONY: check sample validate train merge serve gateway evaluate
