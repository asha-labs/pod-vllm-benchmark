# LLM Benchmark Pod Template

This image runs vLLM benchmarks sequentially for a list of Hugging Face LLMs.

## Build

```bash
docker build -t vllm-llm-bench:latest .
```

## Default command

The image runs:

```bash
python /app/bench_llms.py
```

## Override example (Runpod Start Command)

```bash
python /app/bench_llms.py \
  --models "Qwen/Qwen3-Coder-Next,Qwen/Qwen3-Coder-Next-FP8" \
  --length-pairs "128:128,512:128,1024:256" \
  --gpu-memory-utilization 0.95
```

## Key env vars

- `MODELS` (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `LENGTH_PAIRS` (default: `128:128`)
- `MODELS_DIR` (default: `/models`)
- `GPU_MEMORY_UTILIZATION` (default: `0.95`)
- `MAX_MODEL_LEN` (default: `8128`)
- `DTYPE` (default: `bfloat16`)
- `VLLM_EXTRA_ARGS` (default: empty)

Models are downloaded under `MODELS_DIR` and deleted after each run unless `CLEAR_MODEL_AFTER=0`.
