#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import List


WORD_BANK = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "amber", "bison", "cedar", "dingo", "ember",
    "fjord", "glade", "harbor", "ivory", "jungle", "karma", "lunar", "mango",
    "nectar", "onyx", "prairie", "quartz", "raven", "sable", "topaz", "umber",
    "velvet", "willow", "xenon", "yellow", "zenith",
]


def parse_int_list(value: str) -> List[int]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def build_text(rng: random.Random, word_count: int, prefix: str) -> str:
    if word_count <= 1:
        return prefix
    words = [prefix]
    while len(words) < word_count:
        words.append(rng.choice(WORD_BANK))
    return " ".join(words[:word_count])


def generate_dataset(
    output_path: Path,
    num_samples: int,
    query_lengths: List[int],
    doc_lengths: List[int],
    positive_fraction: float,
    seed: int,
) -> int:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not query_lengths:
        query_lengths = [32]
    if not doc_lengths:
        doc_lengths = [128]

    num_samples = max(0, int(num_samples))
    positive_fraction = max(0.0, min(1.0, float(positive_fraction)))
    num_positive = int(round(num_samples * positive_fraction))
    labels = [1] * num_positive + [0] * max(0, num_samples - num_positive)
    rng.shuffle(labels)

    lines_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, label in enumerate(labels):
            q_len = query_lengths[idx % len(query_lengths)]
            d_len = doc_lengths[idx % len(doc_lengths)]

            query_text = build_text(rng, q_len, f"query{idx}")
            if label:
                doc_text = build_text(rng, d_len, f"pos{idx}")
            else:
                doc_text = build_text(rng, d_len, f"neg{idx}")

            payload = {
                "query": query_text,
                "document": doc_text,
                "label": float(label),
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            lines_written += 1

    return lines_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic pairwise reranker dataset for HF finetuning."
    )
    parser.add_argument(
        "--output",
        default="bench_dataset.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of pair samples to generate.",
    )
    parser.add_argument(
        "--query-words",
        type=int,
        default=16,
        help="Query word count if --query-lengths is not provided.",
    )
    parser.add_argument(
        "--doc-words",
        type=int,
        default=128,
        help="Doc word count if --doc-lengths is not provided.",
    )
    parser.add_argument(
        "--query-lengths",
        default="",
        help="Comma-separated query word counts (e.g., 16,32,64).",
    )
    parser.add_argument(
        "--doc-lengths",
        default="",
        help="Comma-separated doc word counts (e.g., 64,128,256).",
    )
    parser.add_argument(
        "--positive-fraction",
        type=float,
        default=0.5,
        help="Fraction of samples labeled positive (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )

    args = parser.parse_args()

    query_lengths = parse_int_list(args.query_lengths) or [args.query_words]
    doc_lengths = parse_int_list(args.doc_lengths) or [args.doc_words]

    output_path = Path(args.output)
    count = generate_dataset(
        output_path=output_path,
        num_samples=args.num_samples,
        query_lengths=query_lengths,
        doc_lengths=doc_lengths,
        positive_fraction=args.positive_fraction,
        seed=args.seed,
    )

    print(f"Generated {count} samples at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
