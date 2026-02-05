#!/usr/bin/env python3
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T.*?Z\s+")
GPU_RE = re.compile(r"^\|\s*\d+\s+(.+?)\s{2,}(?:On|Off)\s+\|")

LLM_BENCH_RE = re.compile(
    r"Benchmarking\s+(.+?)\s+@\s+in=(\d+),\s+out=(\d+),\s+concurrency=(\d+)"
)
REQ_RE = re.compile(r"Request throughput \(req/s\):\s+([0-9.]+)")
OUT_TOK_RE = re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)")
TOTAL_TOK_RE = re.compile(r"Total token throughput \(tok/s\):\s+([0-9.]+)")
SUCCESS_RE = re.compile(r"Successful requests:\s+(\d+)")
FAILED_RE = re.compile(r"Failed requests:\s+(\d+)")
GEN_TOK_RE = re.compile(r"Total generated tokens:\s+(\d+)")

RERANKER_SUMMARY_RE = re.compile(
    r"OK \| (.+?) \| bs=(\d+) \| ([0-9.]+) pairs/sec \| (\d+) pairs"
)


def clean_line(line: str) -> str:
    line = line.rstrip("\n")
    line = ANSI_RE.sub("", line)
    line = TS_RE.sub("", line)
    return line.strip()


def parse_logs(log_paths: List[Path]) -> Tuple[List[Dict], List[Dict]]:
    llm_results: List[Dict] = []
    reranker_results: List[Dict] = []

    for path in log_paths:
        current_gpu: Optional[str] = None
        current_device: Optional[str] = None
        current_bench: Optional[Dict] = None
        current_req: Optional[str] = None
        current_out: Optional[str] = None
        current_total: Optional[str] = None
        current_success: Optional[int] = None
        current_failed: Optional[int] = None
        current_generated: Optional[int] = None

        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            continue

        for raw_line in lines:
            line = clean_line(raw_line)
            if not line:
                continue

            match = GPU_RE.match(line)
            if match:
                current_gpu = match.group(1).strip()
                continue

            if line.startswith("Device:"):
                device = line.split(":", 1)[1].strip()
                if device:
                    current_device = device
                continue

            match = LLM_BENCH_RE.search(line)
            if match:
                current_bench = {
                    "gpu": current_gpu,
                    "device": current_device,
                    "model": match.group(1).strip(),
                    "input_len": int(match.group(2)),
                    "output_len": int(match.group(3)),
                    "concurrency": int(match.group(4)),
                }
                current_req = None
                current_out = None
                current_total = None
                current_success = None
                current_failed = None
                current_generated = None
                continue

            match = RERANKER_SUMMARY_RE.search(line)
            if match:
                reranker_results.append(
                    {
                        "gpu": current_gpu,
                        "device": current_device,
                        "model": match.group(1).strip(),
                        "batch_size": int(match.group(2)),
                        "pairs_per_sec": float(match.group(3)),
                        "pairs": int(match.group(4)),
                    }
                )
                continue

            if current_bench is None:
                continue

            match = SUCCESS_RE.search(line)
            if match:
                current_success = int(match.group(1))

            match = FAILED_RE.search(line)
            if match:
                current_failed = int(match.group(1))

            match = GEN_TOK_RE.search(line)
            if match:
                current_generated = int(match.group(1))

            match = REQ_RE.search(line)
            if match:
                current_req = match.group(1)

            match = OUT_TOK_RE.search(line)
            if match:
                current_out = match.group(1)

            match = TOTAL_TOK_RE.search(line)
            if match:
                current_total = match.group(1)
                llm_results.append(
                    {
                        **current_bench,
                        "req_s": current_req or "0.00",
                        "out_tok_s": current_out or "0.00",
                        "total_tok_s": current_total,
                        "successful_requests": current_success,
                        "failed_requests": current_failed,
                        "generated_tokens": current_generated,
                    }
                )
                current_bench = None
                current_req = None
                current_out = None
                current_total = None
                current_success = None
                current_failed = None
                current_generated = None

    return llm_results, reranker_results


def fill_missing_gpu(entries: List[Dict], inferred: Optional[str] = None) -> None:
    if inferred is None:
        known = {entry["gpu"] for entry in entries if entry.get("gpu")}
        if len(known) == 1:
            inferred = next(iter(known))
    for entry in entries:
        if entry.get("gpu"):
            continue
        if inferred is not None:
            entry["gpu"] = inferred
            continue
        if entry.get("device"):
            entry["gpu"] = entry["device"]


def dedup_llm(entries: List[Dict]) -> List[Dict]:
    seen = set()
    unique: List[Dict] = []
    for entry in entries:
        key = (
            entry.get("gpu") or "",
            entry["model"],
            entry["input_len"],
            entry["output_len"],
            entry["concurrency"],
            entry["req_s"],
            entry["out_tok_s"],
            entry["total_tok_s"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def dedup_rerankers(entries: List[Dict]) -> List[Dict]:
    seen = set()
    unique: List[Dict] = []
    for entry in entries:
        key = (
            entry.get("gpu") or "",
            entry["model"],
            entry["batch_size"],
            entry["pairs_per_sec"],
            entry["pairs"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def gpu_sort_key(name: Optional[str]) -> Tuple[int, str]:
    if not name or name.lower() == "unknown":
        return (1, "")
    return (0, name)


def filter_llm(
    entries: List[Dict], include_zero: bool, include_failed: bool
) -> List[Dict]:
    filtered: List[Dict] = []
    for entry in entries:
        failed = entry.get("failed_requests") or 0
        if not include_failed and failed > 0:
            continue
        if not include_zero:
            try:
                req_s = float(entry.get("req_s", "0.00"))
                total_tok_s = float(entry.get("total_tok_s", "0.00"))
                out_tok_s = float(entry.get("out_tok_s", "0.00"))
            except ValueError:
                req_s = 0.0
                total_tok_s = 0.0
                out_tok_s = 0.0
            if req_s <= 0.0 or total_tok_s <= 0.0:
                continue
            generated = entry.get("generated_tokens")
            if generated == 0 and out_tok_s <= 0.0:
                continue
        filtered.append(entry)
    return filtered


def aggregate_rerankers(entries: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int, int], List[float]] = defaultdict(list)
    for entry in entries:
        gpu = entry.get("gpu") or "Unknown"
        key = (gpu, entry["model"], entry["batch_size"], entry["pairs"])
        grouped[key].append(float(entry["pairs_per_sec"]))

    aggregated: List[Dict] = []
    for (gpu, model, batch_size, pairs), values in grouped.items():
        avg = mean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
        aggregated.append(
            {
                "gpu": gpu,
                "model": model,
                "batch_size": batch_size,
                "pairs": pairs,
                "runs": len(values),
                "pairs_per_sec_avg": avg,
                "pairs_per_sec_min": min(values),
                "pairs_per_sec_max": max(values),
                "pairs_per_sec_std": std,
            }
        )
    return aggregated


def render_llm_section(entries: List[Dict]) -> List[str]:
    lines: List[str] = []
    lines.append("## LLM Benchmarks")

    if not entries:
        lines.append("")
        lines.append("_No LLM benchmark results found._")
        return lines

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        gpu = entry.get("gpu") or "Unknown"
        grouped[gpu].append(entry)

    for gpu in sorted(grouped.keys(), key=gpu_sort_key):
        lines.append("")
        lines.append(f"### GPU: {gpu}")
        lines.append("")
        lines.append(
            "| Model | Input | Output | Concurrency | Req/s | Tok/s (out) | Tok/s (total) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")

        rows = sorted(
            grouped[gpu],
            key=lambda e: (
                e["model"],
                e["input_len"],
                e["output_len"],
                e["concurrency"],
            ),
        )
        for entry in rows:
            lines.append(
                "| {model} | {input_len} | {output_len} | {concurrency} | {req_s} | {out_tok_s} | {total_tok_s} |".format(
                    **entry
                )
            )

    return lines


def render_reranker_section(entries: List[Dict], aggregate: bool) -> List[str]:
    lines: List[str] = []
    lines.append("")
    lines.append("## Reranker Benchmarks")

    if not entries:
        lines.append("")
        lines.append("_No reranker benchmark results found._")
        return lines

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        gpu = entry.get("gpu") or "Unknown"
        grouped[gpu].append(entry)

    for gpu in sorted(grouped.keys(), key=gpu_sort_key):
        lines.append("")
        lines.append(f"### GPU: {gpu}")
        lines.append("")
        if aggregate:
            lines.append(
                "| Model | Batch size | Pairs | Runs | Pairs/sec avg | Min | Max | Std |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        else:
            lines.append("| Model | Batch size | Pairs/sec | Pairs |")
            lines.append("|---|---:|---:|---:|")

        rows = sorted(
            grouped[gpu],
            key=lambda e: (
                e["model"],
                e["batch_size"],
            ),
        )
        for entry in rows:
            if aggregate:
                lines.append(
                    "| {model} | {batch_size} | {pairs} | {runs} | {avg:.2f} | {min:.2f} | {max:.2f} | {std:.2f} |".format(
                        model=entry["model"],
                        batch_size=entry["batch_size"],
                        pairs=entry["pairs"],
                        runs=entry["runs"],
                        avg=entry["pairs_per_sec_avg"],
                        min=entry["pairs_per_sec_min"],
                        max=entry["pairs_per_sec_max"],
                        std=entry["pairs_per_sec_std"],
                    )
                )
            else:
                lines.append(
                    "| {model} | {batch_size} | {pairs_per_sec:.2f} | {pairs} |".format(
                        model=entry["model"],
                        batch_size=entry["batch_size"],
                        pairs_per_sec=entry["pairs_per_sec"],
                        pairs=entry["pairs"],
                    )
                )

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown benchmark report from log files."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing log files (default: logs).",
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.md",
        help="Output Markdown file path (default: benchmark_report.md).",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include LLM results with zero throughput.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include LLM results with failed requests.",
    )
    parser.add_argument(
        "--reranker-raw",
        action="store_true",
        help="Show raw reranker runs instead of aggregated stats.",
    )

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return 1

    log_paths = sorted([p for p in logs_dir.iterdir() if p.is_file()])
    llm_results, reranker_results = parse_logs(log_paths)

    all_entries = llm_results + reranker_results
    global_known = {entry["gpu"] for entry in all_entries if entry.get("gpu")}
    inferred_gpu = next(iter(global_known)) if len(global_known) == 1 else None

    fill_missing_gpu(llm_results, inferred_gpu)
    fill_missing_gpu(reranker_results, inferred_gpu)

    llm_unique = dedup_llm(llm_results)
    reranker_dedup = dedup_rerankers(reranker_results)

    llm_filtered = filter_llm(
        llm_unique, include_zero=args.include_zero, include_failed=args.include_failed
    )
    if args.reranker_raw:
        reranker_view = reranker_dedup
    else:
        reranker_view = aggregate_rerankers(reranker_results)

    lines: List[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
    )
    lines.append(f"Source logs: {len(log_paths)} file(s) in {logs_dir}")
    lines.append("")

    lines.extend(render_llm_section(llm_filtered))
    lines.extend(render_reranker_section(reranker_view, aggregate=not args.reranker_raw))
    lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
