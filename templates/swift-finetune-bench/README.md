# SWIFT Fine-Tune Benchmark Pod Template

This image runs a short ms-swift fine-tuning job (LoRA by default) to measure training throughput.
The Dockerfile follows the Qwen3-Embedding SWIFT guide and includes DeepSpeed + FlashAttention.

## Build

```bash
docker build -t swift-finetune-bench:latest .
```

## Default command

```bash
bash /app/run_bench.sh
```
The wrapper prints `date` and `nvidia-smi` before running the benchmark.

## Default arguments

```bash
python3 /app/bench_swift_finetune.py \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --dataset "/app/data/train.jsonl" \
  --output-dir "/output" \
  --results-file "/output/swift_finetune_benchmark.txt" \
  --task-type "embedding" \
  --model-type "qwen3_emb" \
  --train-type "lora" \
  --loss-type "infonce" \
  --attn-impl "eager" \
  --torch-dtype "bfloat16" \
  --num-train-epochs 1 \
  --split-dataset-ratio 0.0 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --max-length 256 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --target-modules "all-linear" \
  --eval-strategy "no" \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 10 \
  --save-total-limit 1 \
  --save-only-model 1 \
  --dataset-num-proc 1 \
  --dataloader-num-workers 0 \
  --dataloader-drop-last 1 \
  --max-samples 64
```

## Sample output header

```text
Thu Feb  5 01:11:56 UTC 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...                                                              |
+-----------------------------------------------------------------------------+
================================================================================
RUN CONFIG
Model: Qwen/Qwen3-Embedding-0.6B
Dataset: /app/data/train.jsonl
Used dataset: /app/data/train.jsonl
Max samples: 64
...
================================================================================
```

## Override example (Runpod Start Command)

```bash
python3 /app/bench_swift_finetune.py \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --max-samples 128 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 4 \
  --max-length 256
```

## Key env vars

- `MODEL` (default: `Qwen/Qwen3-Embedding-0.6B`)
- `DATASET` (default: `/app/data/train.jsonl`)
- `OUTPUT_DIR` (default: `/output`)
- `RESULTS_FILE` (default: `/output/swift_finetune_benchmark.txt`)
- `MAX_SAMPLES` (default: `64`)
- `NUM_TRAIN_EPOCHS` (default: `1`)
- `PER_DEVICE_TRAIN_BATCH_SIZE` (default: `4`)
- `GRADIENT_ACCUMULATION_STEPS` (default: `1`)
- `MAX_LENGTH` (default: `256`)
- `TRAIN_TYPE` (default: `lora`)
- `TORCH_DTYPE` (default: `bfloat16`)
- `SPLIT_DATASET_RATIO` (default: `0.0`)
- `SWIFT_EXTRA_ARGS` (default: empty)

The script writes a summary line starting with `SWIFT_FT_RESULT` into `RESULTS_FILE` so
`generate_benchmark_report.py` can pick it up.

If you want to use the report generator, set `RESULTS_FILE` to follow this pattern:
`{server_name}_{model_name}_swift_finetune.txt` (replace `/` with `_` in the model name).
