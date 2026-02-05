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
    mode: str,
    listwise_size: int,
    seed: int,
    query_role: str = "user",
    doc_role: str = "user",
) -> int:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not query_lengths:
        query_lengths = [32]
    if not doc_lengths:
        doc_lengths = [128]

    if mode == "pairwise":
        listwise_size = 1
    else:
        listwise_size = max(1, listwise_size)

    lines_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in range(num_samples):
            q_len = query_lengths[idx % len(query_lengths)]
            d_len = doc_lengths[idx % len(doc_lengths)]

            query_text = build_text(rng, q_len, f"query{idx}")
            pos_text = build_text(rng, d_len, f"pos{idx}")

            negatives = []
            for neg_idx in range(listwise_size):
                neg_text = build_text(rng, d_len, f"neg{idx}_{neg_idx}")
                negatives.append([{"role": doc_role, "content": neg_text}])

            payload = {
                "messages": [{"role": query_role, "content": query_text}],
                "positive_messages": [[{"role": doc_role, "content": pos_text}]],
            }
            if negatives:
                payload["negative_messages"] = negatives

            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            lines_written += 1

    return lines_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SWIFT datasets with pairwise or listwise samples."
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
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--query-words",
        type=int,
        default=128,
        help="Query word count if --query-lengths is not provided.",
    )
    parser.add_argument(
        "--doc-words",
        type=int,
        default=256,
        help="Doc word count if --doc-lengths is not provided.",
    )
    parser.add_argument(
        "--query-lengths",
        default="",
        help="Comma-separated query word counts (e.g., 128,256,512).",
    )
    parser.add_argument(
        "--doc-lengths",
        default="",
        help="Comma-separated doc word counts (e.g., 256,512,1024).",
    )
    parser.add_argument(
        "--mode",
        choices=["pairwise", "listwise"],
        default="pairwise",
        help="pairwise (1:1) or listwise (1:N with negatives).",
    )
    parser.add_argument(
        "--listwise-size",
        type=int,
        default=4,
        help="Number of negatives per query for listwise mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )
    parser.add_argument(
        "--query-role",
        default="user",
        help="Role used for the query message (default: user).",
    )
    parser.add_argument(
        "--doc-role",
        default="user",
        help="Role used for positive/negative documents (default: user).",
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
        mode=args.mode,
        listwise_size=args.listwise_size,
        seed=args.seed,
        query_role=args.query_role,
        doc_role=args.doc_role,
    )

    print(f"Generated {count} samples at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
