#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from generate_swift_dataset import generate_dataset


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_list(value: str) -> List[int]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def parse_csv(value: str) -> List[str]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def infer_task_type(model: str, requested: str) -> str:
    if requested != "embedding":
        return requested
    lower = model.lower()
    if "reranker" in lower:
        if "qwen3-reranker" in lower or "qwen3-vl-reranker" in lower:
            return "generative_reranker"
        return "reranker"
    return requested


def infer_loss_type(task_type: str, requested: str) -> str:
    if requested != "infonce":
        return requested
    if "reranker" in task_type:
        return "pointwise_reranker"
    return requested


def infer_model_type(model: str, requested: str, task_type: str) -> str:
    if requested and requested != "qwen3_emb":
        return requested
    lower = model.lower()
    if "modernbert" in lower and task_type == "reranker":
        if "gte" in lower:
            return "modern_bert_gte_reranker"
        return "modern_bert"
    return requested


def normalize_model_id(model: str) -> str:
    if model.startswith("Alibaba-NLP/gte-reranker-modernbert-base"):
        return "iic/gte-reranker-modernbert-base"
    return model


def detect_nproc_per_node() -> int:
    env_nproc = os.environ.get("NPROC_PER_NODE")
    if env_nproc:
        try:
            return max(1, int(env_nproc))
        except ValueError:
            pass
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("GPU_IDS") or ""
    if raw.strip():
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return max(1, len(parts))
    return 1


def count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def limit_dataset(src: Path, dst: Path, max_samples: int) -> int:
    written = 0
    with src.open("r", encoding="utf-8") as reader, dst.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            writer.write(line)
            written += 1
            if written >= max_samples:
                break
    return written


def yaml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    return json.dumps(str(value))


def write_config(path: Path, config: dict) -> None:
    lines = [f"{key}: {yaml_value(val)}" for key, val in config.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short ms-swift fine-tuning benchmark.")
    default_models_env = os.environ.get("MODELS") or os.environ.get("RERANKER_MODELS") or ""
    parser.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    parser.add_argument("--models", default=default_models_env)
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "auto"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    parser.add_argument("--results-file", default=os.environ.get("RESULTS_FILE", "swift_finetune_benchmark.txt"))

    parser.add_argument("--task-type", default=os.environ.get("TASK_TYPE", "embedding"))
    parser.add_argument("--model-type", default=os.environ.get("MODEL_TYPE", "qwen3_emb"))
    parser.add_argument("--train-type", default=os.environ.get("TRAIN_TYPE", "lora"))
    parser.add_argument("--loss-type", default=os.environ.get("LOSS_TYPE", "infonce"))
    parser.add_argument("--attn-impl", default=os.environ.get("ATTN_IMPL", "eager"))
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"))

    parser.add_argument("--num-train-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")))
    parser.add_argument("--split-dataset-ratio", type=float, default=float(os.environ.get("SPLIT_DATASET_RATIO", "0.0")))
    parser.add_argument("--per-device-train-batch-size", type=int, default=int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "4")))
    parser.add_argument("--per-device-eval-batch-size", type=int, default=int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "4")))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "1e-4")))
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "256")))

    parser.add_argument("--lora-rank", type=int, default=int(os.environ.get("LORA_RANK", "8")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    parser.add_argument("--target-modules", default=os.environ.get("TARGET_MODULES", "all-linear"))

    parser.add_argument("--eval-strategy", default=os.environ.get("EVAL_STRATEGY", "no"))
    parser.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "1000")))
    parser.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "1000")))
    parser.add_argument("--logging-steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "10")))
    parser.add_argument("--save-total-limit", type=int, default=int(os.environ.get("SAVE_TOTAL_LIMIT", "1")))
    parser.add_argument("--save-only-model", default=os.environ.get("SAVE_ONLY_MODEL", "1"))

    parser.add_argument("--dataset-num-proc", type=int, default=int(os.environ.get("DATASET_NUM_PROC", "1")))
    parser.add_argument("--dataloader-num-workers", type=int, default=int(os.environ.get("DATALOADER_NUM_WORKERS", "0")))
    parser.add_argument("--dataloader-drop-last", default=os.environ.get("DATALOADER_DROP_LAST", "1"))

    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("MAX_SAMPLES", "64")))
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "256")))
    parser.add_argument("--query-words", type=int, default=int(os.environ.get("QUERY_WORDS", "128")))
    parser.add_argument("--doc-words", type=int, default=int(os.environ.get("DOC_WORDS", "256")))
    parser.add_argument("--query-lengths", default=os.environ.get("QUERY_LENGTHS", ""))
    parser.add_argument("--doc-lengths", default=os.environ.get("DOC_LENGTHS", ""))
    parser.add_argument("--sample-mode", default=os.environ.get("SAMPLE_MODE", "pairwise"))
    parser.add_argument("--listwise-size", type=int, default=int(os.environ.get("LISTWISE_SIZE", "4")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "13")))
    parser.add_argument("--swift-extra-args", default=os.environ.get("SWIFT_EXTRA_ARGS", ""))
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")

    args = parser.parse_args()

    models = parse_csv(args.models) if args.models else []
    if not models:
        models = [args.model]

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    nproc_per_node = detect_nproc_per_node()
    os.environ["NPROC_PER_NODE"] = str(nproc_per_node)

    drop_last = parse_bool(args.dataloader_drop_last)
    save_only_model = parse_bool(args.save_only_model)

    overall_return = 0

    for index, model in enumerate(models, start=1):
        model = normalize_model_id(model)
        model_output_dir = base_output_dir
        if len(models) > 1:
            model_output_dir = base_output_dir / sanitize_model_name(model)
        model_output_dir.mkdir(parents=True, exist_ok=True)

        task_type = infer_task_type(model, args.task_type)
        loss_type = infer_loss_type(task_type, args.loss_type)
        model_type = infer_model_type(model, args.model_type, task_type)
        doc_role = "assistant" if "reranker" in task_type else "user"

        results_path = Path(args.results_file)
        if not results_path.is_absolute():
            results_path = model_output_dir / results_path
        elif len(models) > 1:
            suffix = "_" + sanitize_model_name(model)
            if results_path.suffix:
                results_path = results_path.with_name(results_path.stem + suffix + results_path.suffix)
            else:
                results_path = results_path.with_name(results_path.name + suffix)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        log_path = model_output_dir / "swift_finetune.log"

        dataset_path = args.dataset
        dataset_file = Path(dataset_path) if dataset_path not in ("", "auto") else None
        used_dataset_path = dataset_path
        total_samples = None
        effective_samples = None
        generated_dataset = False
        query_lengths = None
        doc_lengths = None

        if dataset_path in ("", "auto"):
            generated_dataset = True
        elif dataset_file and dataset_file.is_file():
            total_samples = count_nonempty_lines(dataset_file)
            effective_samples = total_samples
            if args.max_samples > 0:
                effective_samples = min(total_samples, args.max_samples)
                if effective_samples < total_samples:
                    limited_path = model_output_dir / "bench_dataset.jsonl"
                    effective_samples = limit_dataset(dataset_file, limited_path, effective_samples)
                    used_dataset_path = str(limited_path)
        else:
            if args.max_samples > 0:
                print("[swift-ft] Warning: dataset is not a local file; MAX_SAMPLES cannot be enforced.")

        query_lengths_str = args.query_lengths
        doc_lengths_str = args.doc_lengths

        lengths_scaled = False
        effective_listwise_size = args.listwise_size if args.sample_mode.lower() == "listwise" else 1
        if generated_dataset:
            query_lengths = parse_int_list(args.query_lengths) or [args.query_words]
            doc_lengths = parse_int_list(args.doc_lengths) or [args.doc_words]
            sample_mode = args.sample_mode.lower()
            if sample_mode not in {"pairwise", "listwise"}:
                raise ValueError(f"Invalid sample mode: {args.sample_mode}")
            doc_multiplier = 1 + (effective_listwise_size if sample_mode == "listwise" else 0)
            if args.max_length > 0:
                max_total_words = max(1, int(args.max_length * 0.8))
                total_words = max(query_lengths) + max(doc_lengths) * doc_multiplier
                if total_words > max_total_words:
                    scale = max_total_words / total_words
                    query_lengths = [max(1, int(math.floor(x * scale))) for x in query_lengths]
                    doc_lengths = [max(1, int(math.floor(x * scale))) for x in doc_lengths]
                    lengths_scaled = True
                    print(
                        "[swift-ft] Warning: requested query/doc lengths exceed max_length "
                        f"{args.max_length}; scaling lengths to fit."
                    )
            query_lengths_str = ",".join(str(x) for x in query_lengths)
            doc_lengths_str = ",".join(str(x) for x in doc_lengths)
            used_dataset_path = str(model_output_dir / "bench_dataset.jsonl")
            effective_samples = args.num_samples
            total_samples = args.num_samples
            generate_dataset(
                output_path=Path(used_dataset_path),
                num_samples=args.num_samples,
                query_lengths=query_lengths,
                doc_lengths=doc_lengths,
                mode=sample_mode,
                listwise_size=effective_listwise_size,
                seed=args.seed,
                query_role="user",
                doc_role=doc_role,
            )
            max_requested = max(query_lengths + doc_lengths)
            if args.max_length > 0 and max_requested > args.max_length and not lengths_scaled:
                print(
                    f"[swift-ft] Warning: max_length {args.max_length} is below the "
                    f"requested length {max_requested}; inputs will be truncated."
                )

        split_ratio = max(0.0, min(1.0, args.split_dataset_ratio))
        train_samples_per_epoch = None
        if effective_samples is not None:
            train_samples_per_epoch = int(effective_samples * (1.0 - split_ratio))
            if train_samples_per_epoch <= 0:
                train_samples_per_epoch = effective_samples

        config = {
            "model": model,
            "task_type": task_type,
            "model_type": model_type,
            "tuner_type": args.train_type,
            "loss_type": loss_type,
            "attn_impl": args.attn_impl,
            "dataset": used_dataset_path,
            "split_dataset_ratio": split_ratio,
            "output_dir": str(model_output_dir),
            "eval_strategy": args.eval_strategy,
            "eval_steps": args.eval_steps,
            "num_train_epochs": args.num_train_epochs,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "torch_dtype": args.torch_dtype,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "dataset_num_proc": args.dataset_num_proc,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_drop_last": drop_last,
            "save_total_limit": args.save_total_limit,
            "save_only_model": save_only_model,
        }
        if "reranker" in task_type and model_type == "qwen3_emb":
            config.pop("model_type", None)

        config_path = model_output_dir / "swift_finetune_config.yaml"
        write_config(config_path, config)

        cmd = ["swift", "sft", "--config", str(config_path)]
        if args.swift_extra_args:
            cmd.extend(shlex.split(args.swift_extra_args))

        print("=" * 80)
        print("RUN CONFIG")
        if len(models) > 1:
            print(f"Model run: {index}/{len(models)}")
        print(f"Model: {model}")
        print(f"Dataset: {args.dataset}")
        print(f"Used dataset: {used_dataset_path}")
        print(f"Generated dataset: {generated_dataset}")
        print(f"Max samples: {args.max_samples}")
        print(f"Num samples: {args.num_samples}")
        print(f"Query words: {args.query_words}")
        print(f"Doc words: {args.doc_words}")
        print(f"Query lengths: {query_lengths_str or '(none)'}")
        print(f"Doc lengths: {doc_lengths_str or '(none)'}")
        print(f"Sample mode: {args.sample_mode}")
        print(f"Listwise size: {effective_listwise_size}")
        print(f"Seed: {args.seed}")
        print(f"Output dir: {model_output_dir}")
        print(f"Results file: {results_path}")
        print(f"Train type: {args.train_type}")
        print(f"Task type: {task_type}")
        print(f"Model type: {config.get('model_type', '(auto)')}")
        print(f"Loss type: {loss_type}")
        print(f"Attn impl: {args.attn_impl}")
        print(f"Dtype: {args.torch_dtype}")
        print(f"Num train epochs: {args.num_train_epochs}")
        print(f"Split dataset ratio: {split_ratio}")
        print(f"Per device train batch size: {args.per_device_train_batch_size}")
        print(f"Per device eval batch size: {args.per_device_eval_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max length: {args.max_length}")
        print(f"LoRA rank: {args.lora_rank}")
        print(f"LoRA alpha: {args.lora_alpha}")
        print(f"Target modules: {args.target_modules}")
        print(f"Dataset num proc: {args.dataset_num_proc}")
        print(f"Dataloader num workers: {args.dataloader_num_workers}")
        print(f"Dataloader drop last: {drop_last}")
        print(f"Eval strategy: {args.eval_strategy}")
        print(f"Eval steps: {args.eval_steps}")
        print(f"Save steps: {args.save_steps}")
        print(f"Save total limit: {args.save_total_limit}")
        print(f"Save only model: {save_only_model}")
        print(f"NPROC_PER_NODE: {nproc_per_node}")
        print(f"Swift extra args: {args.swift_extra_args or '(none)'}")
        print(f"Dry run: {args.dry_run}")
        print("=" * 80)

        print("[swift-ft] Config written to:", config_path)
        print("[swift-ft] Command:", " ".join(cmd))
        print(f"[swift-ft] NPROC_PER_NODE={nproc_per_node}")

        if args.dry_run:
            continue

        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                log_file.write(line)
            returncode = process.wait()
        elapsed = time.time() - start

        train_samples_total = None
        if train_samples_per_epoch is not None:
            train_samples_total = train_samples_per_epoch * args.num_train_epochs

        steps_total = None
        steps_per_sec = None
        if train_samples_per_epoch is not None:
            denom = args.per_device_train_batch_size * args.gradient_accumulation_steps * nproc_per_node
            if denom > 0:
                steps_per_epoch = math.ceil(train_samples_per_epoch / denom)
                steps_total = steps_per_epoch * args.num_train_epochs
                if elapsed > 0:
                    steps_per_sec = steps_total / elapsed

        samples_per_sec = None
        if train_samples_total is not None and elapsed > 0:
            samples_per_sec = train_samples_total / elapsed

        def fmt_value(value):
            if value is None:
                return "na"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        summary_parts = [
            "SWIFT_FT_RESULT",
            f"model={model}",
            f"train_type={args.train_type}",
            f"task_type={task_type}",
            f"model_type={config.get('model_type', '')}",
            f"loss_type={loss_type}",
            f"dataset={args.dataset}",
            f"used_dataset={used_dataset_path}",
            f"generated_dataset={generated_dataset}",
            f"sample_mode={args.sample_mode}",
            f"listwise_size={effective_listwise_size}",
            f"query_lengths={query_lengths_str or 'na'}",
            f"doc_lengths={doc_lengths_str or 'na'}",
            f"num_samples={args.num_samples}",
            f"seed={args.seed}",
            f"samples={fmt_value(effective_samples)}",
            f"train_samples={fmt_value(train_samples_total)}",
            f"steps={fmt_value(steps_total)}",
            f"seconds={elapsed:.4f}",
            f"samples_per_sec={fmt_value(samples_per_sec)}",
            f"steps_per_sec={fmt_value(steps_per_sec)}",
            f"per_device_train_batch_size={args.per_device_train_batch_size}",
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}",
            f"nproc_per_node={nproc_per_node}",
            f"max_length={args.max_length}",
            f"split_dataset_ratio={split_ratio}",
            f"num_train_epochs={args.num_train_epochs}",
            f"success={returncode == 0}",
        ]
        summary_line = " | ".join(summary_parts)

        print(summary_line)
        results_path.write_text(summary_line + "\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(summary_line + "\n")

        if returncode != 0:
            overall_return = 1
            print(f"[swift-ft] Training failed with exit code {returncode}.")
        print(f"[swift-ft] Summary written to: {results_path}")

    return overall_return


if __name__ == "__main__":
    sys.exit(main())
