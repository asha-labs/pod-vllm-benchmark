#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset, load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

from generate_hf_reranker_dataset import generate_dataset


LABEL_COLUMNS = ("label", "labels", "score", "scores")


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


def write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scale_lengths(query_lengths: List[int], doc_lengths: List[int], max_length: int) -> Tuple[List[int], List[int], bool]:
    if max_length <= 0:
        return query_lengths, doc_lengths, False
    if not query_lengths or not doc_lengths:
        return query_lengths, doc_lengths, False
    max_total_words = max(1, int(max_length * 0.8))
    total_words = max(query_lengths) + max(doc_lengths)
    if total_words <= max_total_words:
        return query_lengths, doc_lengths, False
    scale = max_total_words / total_words
    query_scaled = [max(1, int(math.floor(x * scale))) for x in query_lengths]
    doc_scaled = [max(1, int(math.floor(x * scale))) for x in doc_lengths]
    return query_scaled, doc_scaled, True


def load_dataset_from_path(dataset_path: Path) -> Dataset:
    suffix = dataset_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(dataset_path), split="train")
    if suffix == ".csv":
        return load_dataset("csv", data_files=str(dataset_path), split="train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files=str(dataset_path), split="train")
    if suffix == ".arrow":
        return load_dataset("arrow", data_files=str(dataset_path), split="train")
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def resolve_label_column(dataset: Dataset, requested: str) -> str:
    if requested:
        if requested not in dataset.column_names:
            raise ValueError(f"Label column '{requested}' not found in dataset columns {dataset.column_names}")
        return requested
    for name in LABEL_COLUMNS:
        if name in dataset.column_names:
            return name
    raise ValueError(
        "No label column found. Provide --label-column or use one of: " + ", ".join(LABEL_COLUMNS)
    )


def resolve_text_columns(dataset: Dataset, requested: str, label_column: str) -> List[str]:
    if requested:
        text_columns = parse_csv(requested)
    else:
        text_columns = [col for col in dataset.column_names if col != label_column]
    if len(text_columns) != 2:
        raise ValueError(
            "Expected exactly two text columns. "
            f"Found {len(text_columns)} from {dataset.column_names}."
        )
    for col in text_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Text column '{col}' not found in dataset columns {dataset.column_names}")
    return text_columns


def normalize_dataset(
    dataset: Dataset,
    text_columns: List[str],
    label_column: str,
) -> Tuple[Dataset, List[str], str]:
    keep_columns = text_columns + [label_column]
    dataset = dataset.select_columns(keep_columns)
    if label_column != "label":
        dataset = dataset.rename_column(label_column, "label")
        label_column = "label"
    return dataset, text_columns, label_column


def resolve_precision(torch_dtype: str) -> Tuple[bool, bool, str]:
    import torch

    torch_dtype = (torch_dtype or "").strip().lower()
    if torch_dtype in {"auto", ""}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, False, "bf16"
        if torch.cuda.is_available():
            return False, True, "fp16"
        return False, False, "fp32"
    if torch_dtype in {"bf16", "bfloat16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, False, "bf16"
        if torch.cuda.is_available():
            return False, True, "fp16"
        return False, False, "fp32"
    if torch_dtype in {"fp16", "float16"}:
        if torch.cuda.is_available():
            return False, True, "fp16"
        return False, False, "fp32"
    if torch_dtype in {"fp32", "float32"}:
        return False, False, "fp32"
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a short HF Sentence Transformers reranker finetuning benchmark."
    )
    default_models_env = os.environ.get("MODELS") or os.environ.get("RERANKER_MODELS") or ""
    parser.add_argument("--model", default=os.environ.get("MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    parser.add_argument("--models", default=default_models_env)
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "auto"))
    parser.add_argument("--dataset-config", default=os.environ.get("DATASET_CONFIG", ""))
    parser.add_argument("--dataset-split", default=os.environ.get("DATASET_SPLIT", "train"))
    parser.add_argument("--eval-dataset", default=os.environ.get("EVAL_DATASET", ""))
    parser.add_argument("--eval-dataset-config", default=os.environ.get("EVAL_DATASET_CONFIG", ""))
    parser.add_argument("--eval-dataset-split", default=os.environ.get("EVAL_DATASET_SPLIT", "validation"))
    parser.add_argument("--text-columns", default=os.environ.get("TEXT_COLUMNS", ""))
    parser.add_argument("--label-column", default=os.environ.get("LABEL_COLUMN", ""))

    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    parser.add_argument(
        "--results-file",
        default=os.environ.get("RESULTS_FILE", "hf_reranker_finetune_benchmark.txt"),
    )

    parser.add_argument("--loss-type", default=os.environ.get("LOSS_TYPE", "bce"))
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"))
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=os.environ.get("TRUST_REMOTE_CODE", "0") == "1",
    )

    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")),
    )
    parser.add_argument(
        "--split-dataset-ratio",
        type=float,
        default=float(os.environ.get("SPLIT_DATASET_RATIO", "0.1")),
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "4")),
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "8")),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1")),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("LEARNING_RATE", "2e-5")),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=float(os.environ.get("WARMUP_RATIO", "0.1")),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(os.environ.get("WEIGHT_DECAY", "0.0")),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.environ.get("MAX_LENGTH", "512")),
    )
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "-1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "13")))

    parser.add_argument("--eval-strategy", default=os.environ.get("EVAL_STRATEGY", "steps"))
    parser.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "100")))
    parser.add_argument("--save-strategy", default=os.environ.get("SAVE_STRATEGY", "steps"))
    parser.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "100")))
    parser.add_argument("--logging-steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "10")))
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=int(os.environ.get("SAVE_TOTAL_LIMIT", "1")),
    )
    parser.add_argument("--report-to", default=os.environ.get("REPORT_TO", "none"))
    parser.add_argument("--dataloader-num-workers", type=int, default=int(os.environ.get("DATALOADER_NUM_WORKERS", "0")))
    parser.add_argument("--dataloader-drop-last", default=os.environ.get("DATALOADER_DROP_LAST", "1"))

    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("MAX_SAMPLES", "256")))
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "256")))
    parser.add_argument("--query-words", type=int, default=int(os.environ.get("QUERY_WORDS", "16")))
    parser.add_argument("--doc-words", type=int, default=int(os.environ.get("DOC_WORDS", "128")))
    parser.add_argument("--query-lengths", default=os.environ.get("QUERY_LENGTHS", ""))
    parser.add_argument("--doc-lengths", default=os.environ.get("DOC_LENGTHS", ""))
    parser.add_argument(
        "--positive-fraction",
        type=float,
        default=float(os.environ.get("POSITIVE_FRACTION", "0.5")),
    )
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN", "0") == "1")

    args = parser.parse_args()

    models = parse_csv(args.models) if args.models else []
    if not models:
        models = [args.model]

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    drop_last = parse_bool(args.dataloader_drop_last)
    loss_type = args.loss_type.lower().strip()
    if loss_type not in {"bce", "binary_cross_entropy", "binary_crossentropy"}:
        raise ValueError("Only binary cross entropy loss is supported in this benchmark.")

    overall_return = 0

    for index, model_name in enumerate(models, start=1):
        model_output_dir = base_output_dir
        if len(models) > 1:
            model_output_dir = base_output_dir / sanitize_model_name(model_name)
        model_output_dir.mkdir(parents=True, exist_ok=True)

        results_path = Path(args.results_file)
        if not results_path.is_absolute():
            results_path = model_output_dir / results_path
        elif len(models) > 1:
            suffix = "_" + sanitize_model_name(model_name)
            if results_path.suffix:
                results_path = results_path.with_name(results_path.stem + suffix + results_path.suffix)
            else:
                results_path = results_path.with_name(results_path.name + suffix)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_spec = args.dataset
        dataset_path = Path(dataset_spec) if dataset_spec not in ("", "auto") else None
        used_dataset_path = dataset_spec
        generated_dataset = False
        lengths_scaled = False
        query_lengths = None
        doc_lengths = None

        if dataset_spec in ("", "auto"):
            generated_dataset = True
            query_lengths = parse_int_list(args.query_lengths) or [args.query_words]
            doc_lengths = parse_int_list(args.doc_lengths) or [args.doc_words]
            query_lengths, doc_lengths, lengths_scaled = scale_lengths(
                query_lengths,
                doc_lengths,
                args.max_length,
            )
            used_dataset_path = str(model_output_dir / "bench_dataset.jsonl")
            generate_dataset(
                output_path=Path(used_dataset_path),
                num_samples=args.num_samples,
                query_lengths=query_lengths,
                doc_lengths=doc_lengths,
                positive_fraction=args.positive_fraction,
                seed=args.seed,
            )
            dataset = load_dataset("json", data_files=used_dataset_path, split="train")
        elif dataset_path and dataset_path.exists():
            dataset = load_dataset_from_path(dataset_path)
            used_dataset_path = str(dataset_path)
        else:
            dataset = load_dataset(dataset_spec, args.dataset_config or None, split=args.dataset_split)

        split_ratio = max(0.0, min(1.0, args.split_dataset_ratio))

        total_samples = len(dataset)
        effective_samples = total_samples
        if not generated_dataset and args.max_samples > 0 and total_samples > args.max_samples:
            dataset = dataset.select(range(args.max_samples))
            effective_samples = len(dataset)

        label_column = resolve_label_column(dataset, args.label_column)
        text_columns = resolve_text_columns(dataset, args.text_columns, label_column)

        eval_dataset = None
        eval_dataset_spec = None
        if args.eval_dataset:
            eval_dataset_spec = args.eval_dataset
            eval_dataset_path = Path(args.eval_dataset)
            if eval_dataset_path.exists():
                eval_dataset = load_dataset_from_path(eval_dataset_path)
                eval_dataset_spec = str(eval_dataset_path)
            else:
                eval_dataset = load_dataset(
                    args.eval_dataset,
                    args.eval_dataset_config or None,
                    split=args.eval_dataset_split,
                )
        elif split_ratio > 0:
            split = dataset.train_test_split(test_size=split_ratio, seed=args.seed, shuffle=True)
            dataset = split["train"]
            eval_dataset = split["test"]
            effective_samples = len(dataset)

        train_dataset, text_columns, label_column = normalize_dataset(dataset, text_columns, label_column)
        if eval_dataset is not None:
            eval_dataset, _, _ = normalize_dataset(eval_dataset, text_columns, label_column)

        eval_strategy = args.eval_strategy
        if eval_dataset is None and eval_strategy != "no":
            eval_strategy = "no"

        evaluator = None
        if eval_dataset is not None and eval_strategy != "no":
            pairs = list(zip(eval_dataset[text_columns[0]], eval_dataset[text_columns[1]]))
            scores = eval_dataset["label"]
            evaluator = CrossEncoderCorrelationEvaluator(
                sentence_pairs=pairs,
                scores=scores,
                name="eval",
            )

        config = {
            "model": model_name,
            "dataset": dataset_spec,
            "used_dataset": used_dataset_path,
            "generated_dataset": generated_dataset,
            "eval_dataset": eval_dataset_spec or "",
            "text_columns": text_columns,
            "label_column": label_column,
            "loss_type": loss_type,
            "torch_dtype": args.torch_dtype,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "max_steps": args.max_steps,
            "split_dataset_ratio": split_ratio,
            "eval_strategy": eval_strategy,
            "eval_steps": args.eval_steps,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
            "report_to": args.report_to,
            "dataloader_num_workers": args.dataloader_num_workers,
            "dataloader_drop_last": drop_last,
            "seed": args.seed,
        }

        config_path = model_output_dir / "hf_reranker_finetune_config.json"
        write_config(config_path, config)

        print("=" * 80)
        print("RUN CONFIG")
        if len(models) > 1:
            print(f"Model run: {index}/{len(models)}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_spec}")
        print(f"Used dataset: {used_dataset_path}")
        print(f"Generated dataset: {generated_dataset}")
        print(f"Max samples: {args.max_samples}")
        print(f"Effective samples: {effective_samples}")
        print(f"Num samples: {args.num_samples}")
        print(f"Positive fraction: {args.positive_fraction}")
        print(f"Query words: {args.query_words}")
        print(f"Doc words: {args.doc_words}")
        print(f"Query lengths: {','.join(str(x) for x in (query_lengths or [])) or '(none)'}")
        print(f"Doc lengths: {','.join(str(x) for x in (doc_lengths or [])) or '(none)'}")
        if lengths_scaled:
            print("Lengths scaled to fit max_length.")
        print(f"Text columns: {', '.join(text_columns)}")
        print(f"Label column: {label_column}")
        print(f"Eval dataset: {eval_dataset_spec or '(none)'}")
        print(f"Output dir: {model_output_dir}")
        print(f"Results file: {results_path}")
        print(f"Loss type: {loss_type}")
        print(f"Torch dtype: {args.torch_dtype}")
        print(f"Num train epochs: {args.num_train_epochs}")
        print(f"Split dataset ratio: {split_ratio}")
        print(f"Per device train batch size: {args.per_device_train_batch_size}")
        print(f"Per device eval batch size: {args.per_device_eval_batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Warmup ratio: {args.warmup_ratio}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Max length: {args.max_length}")
        print(f"Max steps: {args.max_steps}")
        print(f"Eval strategy: {eval_strategy}")
        print(f"Eval steps: {args.eval_steps}")
        print(f"Save strategy: {args.save_strategy}")
        print(f"Save steps: {args.save_steps}")
        print(f"Logging steps: {args.logging_steps}")
        print(f"Save total limit: {args.save_total_limit}")
        print(f"Report to: {args.report_to}")
        print(f"Dataloader num workers: {args.dataloader_num_workers}")
        print(f"Dataloader drop last: {drop_last}")
        print(f"Seed: {args.seed}")
        print(f"Dry run: {args.dry_run}")
        print("=" * 80)
        print("[hf-ft] Config written to:", config_path)

        if args.dry_run:
            continue

        import torch

        use_bf16, use_fp16, resolved_dtype = resolve_precision(args.torch_dtype)
        if args.torch_dtype and args.torch_dtype.lower() not in {"auto", ""}:
            if args.torch_dtype.lower().startswith("bf") and not use_bf16:
                print("[hf-ft] Warning: bf16 unsupported; falling back to", resolved_dtype)
            if args.torch_dtype.lower().startswith("fp16") and not use_fp16:
                print("[hf-ft] Warning: fp16 unsupported; falling back to", resolved_dtype)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

        training_args = CrossEncoderTrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            fp16=use_fp16,
            bf16=use_bf16,
            eval_strategy=eval_strategy,
            eval_steps=args.eval_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            report_to=args.report_to,
            run_name=sanitize_model_name(model_name),
            remove_unused_columns=False,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_drop_last=drop_last,
            seed=args.seed,
        )

        model = CrossEncoder(
            model_name,
            num_labels=1,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
        )

        loss = BinaryCrossEntropyLoss(model)

        trainer = CrossEncoderTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
        )

        start = time.time()
        success = True
        try:
            trainer.train()
            trainer.save_model()
        except Exception as exc:
            success = False
            overall_return = 1
            print(f"[hf-ft] Training failed: {exc}")
        elapsed = time.time() - start

        steps_total = None
        if trainer.state is not None:
            steps_total = trainer.state.global_step

        train_samples_total = None
        if effective_samples is not None:
            train_samples_total = effective_samples * args.num_train_epochs

        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if n_gpus > 0:
            effective_batch *= n_gpus

        samples_per_sec = None
        if train_samples_total is not None and elapsed > 0:
            samples_per_sec = train_samples_total / elapsed

        steps_per_sec = None
        if steps_total is not None and elapsed > 0:
            steps_per_sec = steps_total / elapsed

        def fmt_value(value):
            if value is None:
                return "na"
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        summary_parts = [
            "HF_RERANKER_FT_RESULT",
            f"model={model_name}",
            f"loss_type={loss_type}",
            f"dataset={dataset_spec}",
            f"used_dataset={used_dataset_path}",
            f"generated_dataset={generated_dataset}",
            f"eval_dataset={eval_dataset_spec or 'na'}",
            f"text_columns={','.join(text_columns)}",
            f"label_column={label_column}",
            f"num_samples={args.num_samples}",
            f"positive_fraction={args.positive_fraction}",
            f"max_samples={args.max_samples}",
            f"samples={fmt_value(effective_samples)}",
            f"train_samples={fmt_value(train_samples_total)}",
            f"steps={fmt_value(steps_total)}",
            f"seconds={elapsed:.4f}",
            f"samples_per_sec={fmt_value(samples_per_sec)}",
            f"steps_per_sec={fmt_value(steps_per_sec)}",
            f"per_device_train_batch_size={args.per_device_train_batch_size}",
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}",
            f"effective_batch_size={effective_batch}",
            f"num_gpus={n_gpus}",
            f"max_length={args.max_length}",
            f"split_dataset_ratio={split_ratio}",
            f"num_train_epochs={args.num_train_epochs}",
            f"torch_dtype={resolved_dtype}",
            f"success={success}",
        ]
        summary_line = " | ".join(summary_parts)

        print(summary_line)
        results_path.write_text(summary_line + "\n", encoding="utf-8")
        print(f"[hf-ft] Summary written to: {results_path}")

    return overall_return


if __name__ == "__main__":
    sys.exit(main())
