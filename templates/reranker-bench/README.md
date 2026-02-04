# Reranker Benchmark Pod Template

This image benchmarks reranker/cross-encoder models using Transformers.

## Build

```bash
docker build -t reranker-bench:latest .
```

## Default command

```bash
python /app/bench_rerankers.py
```

## Override example (Runpod Start Command)

```bash
python3 /app/bench_rerankers.py \
  --models "Qwen/Qwen3-Reranker-Base,cross-encoder/ms-marco-MiniLM-L-6-v2" \
  --num-pairs 2048 \
  --batch-size 32 \
  --max-length 512
```

## Key env vars

- `RERANKER_MODELS` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `MODELS_DIR` (default: `/models`)
- `NUM_PAIRS` (default: `1024`)
- `BATCH_SIZE` (default: `16`)
- `BATCH_SIZE_SWEEP` (optional, e.g. `8,16,32`)
- `NUM_PAIRS_PER_BATCH` (optional, e.g. `64`; if set, pairs scale with batch size)
- `MIN_PAIRS` (optional floor when scaling pairs)
- `MAX_LENGTH` (default: `512`)
- `TRUST_REMOTE_CODE` (default: `1`)

Models are downloaded under `MODELS_DIR` and deleted after each run unless `CLEAR_MODEL_AFTER=0`.
