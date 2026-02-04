#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
from typing import List

from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_csv(value: str) -> List[str]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def parse_int_list(value: str) -> List[int]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def sanitize_cuda_env(env: dict) -> dict:
    compat_path = "/usr/local/cuda/compat"
    ld_path = env.get("LD_LIBRARY_PATH", "")
    if ld_path:
        parts = [p for p in ld_path.split(":") if p and compat_path not in p]
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    env.setdefault("CUDA_DISABLE_COMPAT", "1")
    return env


def download_model(repo_id: str, local_dir: str, token: str) -> None:
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading model: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        cache_dir=local_dir,
        token=token or None,
        resume_download=True,
    )


def make_text(word_count: int) -> str:
    return ("hello " * word_count).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark reranker/cross-encoder models.")
    parser.add_argument(
        "--models",
        default=os.environ.get("RERANKER_MODELS", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        help="Comma-separated Hugging Face model IDs.",
    )
    parser.add_argument("--num-pairs", type=int, default=int(os.environ.get("NUM_PAIRS", "1024")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    parser.add_argument(
        "--batch-size-sweep",
        default=os.environ.get("BATCH_SIZE_SWEEP", ""),
        help="Comma-separated list of batch sizes to sweep (overrides --batch-size).",
    )
    parser.add_argument(
        "--num-pairs-per-batch",
        type=int,
        default=int(os.environ.get("NUM_PAIRS_PER_BATCH", "0")),
        help="If set > 0, num_pairs = max(min_pairs, value * batch_size).",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=int(os.environ.get("MIN_PAIRS", "0")),
        help="Minimum pairs when scaling by batch size.",
    )
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "512")))
    parser.add_argument("--query-words", type=int, default=int(os.environ.get("QUERY_WORDS", "16")))
    parser.add_argument("--doc-words", type=int, default=int(os.environ.get("DOC_WORDS", "128")))
    parser.add_argument("--warmup-steps", type=int, default=int(os.environ.get("WARMUP_STEPS", "2")))
    parser.add_argument("--use-fp16", action="store_true", default=os.environ.get("USE_FP16", "0") == "1")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=os.environ.get("TRUST_REMOTE_CODE", "1") != "0",
    )

    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/models"))
    parser.add_argument("--hf-token", default=os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    parser.add_argument("--clear-model-after", action="store_true", default=os.environ.get("CLEAR_MODEL_AFTER", "1") != "0")
    parser.add_argument("--fail-on-error", action="store_true", default=os.environ.get("FAIL_ON_ERROR", "0") == "1")

    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    hf_home = os.path.join(args.models_dir, ".hf_home")
    os.makedirs(hf_home, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    base_env = sanitize_cuda_env(os.environ.copy())
    os.environ.update(base_env)

    import torch

    models = parse_csv(args.models)
    if not models:
        print("No models provided.")
        return 1

    sweep_values = parse_int_list(args.batch_size_sweep)
    if sweep_values:
        batch_sizes = sweep_values
    else:
        batch_sizes = [args.batch_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    failed_models = []

    for model in models:
        safe_name = sanitize_model_name(model)
        local_dir = os.path.join(args.models_dir, safe_name)

        print("=" * 80)
        print(f"Model: {model}")
        print(f"Local dir: {local_dir}")
        print(f"Device: {device}")
        print("=" * 80)

        try:
            download_model(model, local_dir, args.hf_token)
        except Exception as exc:
            print(f"Download failed for {model}: {exc}")
            failed_models.append((model, "download_failed"))
            if args.clear_model_after:
                shutil.rmtree(local_dir, ignore_errors=True)
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_dir, trust_remote_code=args.trust_remote_code
            )
            model_obj = AutoModelForSequenceClassification.from_pretrained(
                local_dir, trust_remote_code=args.trust_remote_code
            )
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.sep_token is not None:
                    tokenizer.pad_token = tokenizer.sep_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model_obj.resize_token_embeddings(len(tokenizer))
            model_obj.eval()
            model_obj.to(device)
            if args.use_fp16 and device.type == "cuda":
                model_obj.half()
        except Exception as exc:
            print(f"Model load failed for {model}: {exc}")
            failed_models.append((model, "load_failed"))
            if args.clear_model_after:
                shutil.rmtree(local_dir, ignore_errors=True)
            continue

        def forward_batch(encoded):
            try:
                return model_obj(**encoded)
            except TypeError as exc:
                if "token_type_ids" in str(exc) and "unexpected keyword argument" in str(exc):
                    encoded = dict(encoded)
                    encoded.pop("token_type_ids", None)
                    return model_obj(**encoded)
                raise

        query_text = make_text(args.query_words)
        doc_text = make_text(args.doc_words)

        try:
            for batch_size in batch_sizes:
                if args.num_pairs_per_batch > 0:
                    effective_pairs = max(
                        args.min_pairs,
                        args.num_pairs_per_batch * batch_size,
                    )
                else:
                    effective_pairs = args.num_pairs

                pairs = [(query_text, doc_text) for _ in range(effective_pairs)]

                def run_steps(steps: int) -> None:
                    if steps <= 0:
                        return
                    for _ in range(steps):
                        batch = pairs[:batch_size]
                        encoded = tokenizer(
                            [q for q, _ in batch],
                            [d for _, d in batch],
                            padding=True,
                            truncation=True,
                            max_length=args.max_length,
                            return_tensors="pt",
                        )
                        encoded = {k: v.to(device) for k, v in encoded.items()}
                        with torch.no_grad():
                            _ = forward_batch(encoded)
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                if args.warmup_steps > 0:
                    print(f"Running warmup (batch_size={batch_size})...")
                    run_steps(args.warmup_steps)

                print(f"Running benchmark (batch_size={batch_size})...")
                start = time.time()
                processed = 0
                for i in range(0, effective_pairs, batch_size):
                    batch = pairs[i : i + batch_size]
                    encoded = tokenizer(
                        [q for q, _ in batch],
                        [d for _, d in batch],
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                        return_tensors="pt",
                    )
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    with torch.no_grad():
                        _ = forward_batch(encoded)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    processed += len(batch)

                elapsed = time.time() - start
                pairs_per_sec = processed / elapsed if elapsed > 0 else 0.0

                print(f"Processed {processed} pairs in {elapsed:.2f}s")
                print(f"Throughput: {pairs_per_sec:.2f} pairs/sec")

                results.append({
                    "model": model,
                    "batch_size": batch_size,
                    "pairs": processed,
                    "seconds": elapsed,
                    "pairs_per_sec": pairs_per_sec,
                })
        except Exception as exc:
            print(f"Inference failed for {model}: {exc}")
            failed_models.append((model, "inference_failed"))

        if args.clear_model_after:
            shutil.rmtree(local_dir, ignore_errors=True)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for item in results:
        print(
            f"OK | {item['model']} | bs={item['batch_size']} | {item['pairs_per_sec']:.2f} pairs/sec | {item['pairs']} pairs"
        )

    if failed_models:
        print("=" * 80)
        print("FAILED MODELS")
        print("=" * 80)
        for model, reason in failed_models:
            print(f"{model}: {reason}")

    if failed_models and args.fail_on_error:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
