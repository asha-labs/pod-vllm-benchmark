#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_csv(value: str) -> List[str]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


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
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "512")))
    parser.add_argument("--query-words", type=int, default=int(os.environ.get("QUERY_WORDS", "16")))
    parser.add_argument("--doc-words", type=int, default=int(os.environ.get("DOC_WORDS", "128")))
    parser.add_argument("--warmup-steps", type=int, default=int(os.environ.get("WARMUP_STEPS", "2")))
    parser.add_argument("--use-fp16", action="store_true", default=os.environ.get("USE_FP16", "0") == "1")

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

    models = parse_csv(args.models)
    if not models:
        print("No models provided.")
        return 1

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
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model_obj = AutoModelForSequenceClassification.from_pretrained(local_dir)
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

        query_text = make_text(args.query_words)
        doc_text = make_text(args.doc_words)
        pairs = [(query_text, doc_text) for _ in range(args.num_pairs)]

        def run_steps(steps: int) -> None:
            if steps <= 0:
                return
            for _ in range(steps):
                batch = pairs[: args.batch_size]
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
                    _ = model_obj(**encoded)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        if args.warmup_steps > 0:
            print("Running warmup...")
            run_steps(args.warmup_steps)

        print("Running benchmark...")
        start = time.time()
        processed = 0
        for i in range(0, args.num_pairs, args.batch_size):
            batch = pairs[i : i + args.batch_size]
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
                _ = model_obj(**encoded)
            if device.type == "cuda":
                torch.cuda.synchronize()
            processed += len(batch)

        elapsed = time.time() - start
        pairs_per_sec = processed / elapsed if elapsed > 0 else 0.0

        print(f"Processed {processed} pairs in {elapsed:.2f}s")
        print(f"Throughput: {pairs_per_sec:.2f} pairs/sec")

        results.append({
            "model": model,
            "pairs": processed,
            "seconds": elapsed,
            "pairs_per_sec": pairs_per_sec,
        })

        if args.clear_model_after:
            shutil.rmtree(local_dir, ignore_errors=True)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for item in results:
        print(
            f"OK | {item['model']} | {item['pairs_per_sec']:.2f} pairs/sec | {item['pairs']} pairs"
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
