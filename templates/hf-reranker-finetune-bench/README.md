# HF Reranker Fine-tune Benchmark Pod Template

This image runs a short Sentence Transformers v4 CrossEncoder fine-tuning job to measure
training throughput. It follows the Hugging Face reranker training guide and uses
`CrossEncoderTrainer` with `BinaryCrossEntropyLoss` on pairwise data.

## Build

```bash
docker build -t hf-reranker-finetune-bench:latest .
```

## Default command

```bash
bash /app/run_bench.sh
```

The wrapper prints `date` and `nvidia-smi` before running the benchmark.

## Default arguments

```bash
python3 /app/bench_hf_reranker_finetune.py \
  --models "" \
  --model "cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --dataset "auto" \
  --dataset-config "" \
  --dataset-split "train" \
  --eval-dataset "" \
  --eval-dataset-config "" \
  --eval-dataset-split "validation" \
  --text-columns "" \
  --label-column "" \
  --output-dir "/output" \
  --results-file "hf_reranker_finetune_benchmark.txt" \
  --loss-type "bce" \
  --torch-dtype "bfloat16" \
  --num-train-epochs 1 \
  --split-dataset-ratio 0.1 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 1 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --weight-decay 0.0 \
  --max-length 512 \
  --max-steps -1 \
  --seed 13 \
  --eval-strategy "steps" \
  --eval-steps 100 \
  --save-strategy "steps" \
  --save-steps 100 \
  --logging-steps 10 \
  --save-total-limit 1 \
  --report-to "none" \
  --dataloader-num-workers 0 \
  --dataloader-drop-last 1 \
  --max-samples 256 \
  --num-samples 256 \
  --query-words 16 \
  --doc-words 128 \
  --query-lengths "" \
  --doc-lengths "" \
  --positive-fraction 0.5
```

## Sample output header

```text
Thu Feb  5 01:11:56 UTC 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...                                                              |
+-----------------------------------------------------------------------------+
================================================================================
RUN CONFIG
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
Dataset: auto
Used dataset: /output/bench_dataset.jsonl
Generated dataset: True
Max samples: 256
...
================================================================================
```

## Dataset format

The trainer expects two text columns and one label column. For `BinaryCrossEntropyLoss`,
labels should be 0/1 (or floats in [0, 1]). By default, the script looks for a label
column named `label`, `labels`, `score`, or `scores` and treats the remaining two columns
as inputs. If your dataset uses different column names, set `TEXT_COLUMNS` and
`LABEL_COLUMN`.

## Dataset generator

The benchmark generates a synthetic pairwise dataset on the fly. You can also call the
generator directly:

```bash
python3 /app/generate_hf_reranker_dataset.py \
  --output /output/bench_dataset.jsonl \
  --num-samples 512 \
  --query-lengths 16,32,64 \
  --doc-lengths 64,128,256 \
  --positive-fraction 0.5
```

## Override example (Runpod Start Command)

```bash
python3 /app/bench_hf_reranker_finetune.py \
  --models "cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base" \
  --dataset auto \
  --num-samples 512 \
  --query-lengths 32,64 \
  --doc-lengths 128,256 \
  --per-device-train-batch-size 8 \
  --max-length 256
```

## Env var example

```bash
DATASET=auto \
QUERY_LENGTHS=32,64 \
DOC_LENGTHS=128,256 \
POSITIVE_FRACTION=0.5 \
/bin/bash -lc 'nvidia-smi && python3 /app/bench_hf_reranker_finetune.py'
```

## Key env vars

- `MODELS` (comma-separated; overrides `MODEL`)
- `RERANKER_MODELS` (alias for `MODELS` if `MODELS` is unset)
- `MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `DATASET` (default: `auto`)
- `DATASET_CONFIG` (HF dataset config; default empty)
- `DATASET_SPLIT` (default: `train`)
- `EVAL_DATASET` (optional separate eval dataset)
- `EVAL_DATASET_CONFIG` (eval config; default empty)
- `EVAL_DATASET_SPLIT` (default: `validation`)
- `TEXT_COLUMNS` (e.g. `query,document`)
- `LABEL_COLUMN` (e.g. `label` or `score`)
- `OUTPUT_DIR` (default: `/output`)
- `RESULTS_FILE` (default: `hf_reranker_finetune_benchmark.txt`)
- `LOSS_TYPE` (default: `bce`; currently fixed to BCE)
- `TRUST_REMOTE_CODE` (default: `0`)
- `MAX_SAMPLES` (default: `256`)
- `NUM_SAMPLES` (default: `256`)
- `QUERY_WORDS` (default: `16`)
- `DOC_WORDS` (default: `128`)
- `QUERY_LENGTHS` (default: empty)
- `DOC_LENGTHS` (default: empty)
- `POSITIVE_FRACTION` (default: `0.5`)
- `NUM_TRAIN_EPOCHS` (default: `1`)
- `SPLIT_DATASET_RATIO` (default: `0.1`)
- `PER_DEVICE_TRAIN_BATCH_SIZE` (default: `4`)
- `PER_DEVICE_EVAL_BATCH_SIZE` (default: `8`)
- `GRADIENT_ACCUMULATION_STEPS` (default: `1`)
- `LEARNING_RATE` (default: `2e-5`)
- `WARMUP_RATIO` (default: `0.1`)
- `WEIGHT_DECAY` (default: `0.0`)
- `MAX_LENGTH` (default: `512`)
- `MAX_STEPS` (default: `-1`)
- `TORCH_DTYPE` (default: `bfloat16`; auto-falls back if unsupported)
- `EVAL_STRATEGY` (default: `steps`)
- `EVAL_STEPS` (default: `100`)
- `SAVE_STRATEGY` (default: `steps`)
- `SAVE_STEPS` (default: `100`)
- `LOGGING_STEPS` (default: `10`)
- `SAVE_TOTAL_LIMIT` (default: `1`)
- `REPORT_TO` (default: `none`)
- `DATALOADER_NUM_WORKERS` (default: `0`)
- `DATALOADER_DROP_LAST` (default: `1`)
- `SEED` (default: `13`)

When `MODELS` is set with multiple entries, each run is written under
`$OUTPUT_DIR/<model_name>` with separate logs/results.

The script writes a summary line starting with `HF_RERANKER_FT_RESULT` into `RESULTS_FILE`.
