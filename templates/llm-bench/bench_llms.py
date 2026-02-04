#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import List, Tuple

from huggingface_hub import snapshot_download


def parse_csv(value: str) -> List[str]:
    items = []
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            items.append(part)
    return items


def parse_length_pairs(value: str) -> List[Tuple[int, int]]:
    pairs = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            left, right = part.split(":", 1)
        elif "x" in part:
            left, right = part.split("x", 1)
        else:
            raise ValueError(f"Invalid length pair '{part}'. Use IN:OUT (e.g., 128:128).")
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs


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


def wait_for_health(proc: subprocess.Popen, url: str, timeout_s: int) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            print("vLLM process exited before becoming healthy.")
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    print(f"Timed out waiting for health endpoint after {timeout_s}s")
    return False


def run_bench(
    model_name: str,
    base_url: str,
    input_len: int,
    output_len: int,
    max_concurrency: int,
    num_prompts: int,
    api_key: str,
    env: dict,
) -> Tuple[bool, str]:
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model_name,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--max-concurrency",
        str(max_concurrency),
        "--num-prompts",
        str(num_prompts),
        "--ignore-eos",
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--base-url",
        base_url,
    ]

    env = env.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = (result.stdout or "") + (result.stderr or "")
    print(output)
    return result.returncode == 0, output


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark multiple LLMs with vLLM.")
    parser.add_argument(
        "--models",
        default=os.environ.get("MODELS", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        help="Comma-separated Hugging Face model IDs.",
    )
    parser.add_argument(
        "--length-pairs",
        default=os.environ.get("LENGTH_PAIRS", "128:128"),
        help="Comma-separated input:output length pairs (e.g., 128:128,512:128).",
    )
    parser.add_argument("--max-concurrency", type=int, default=int(os.environ.get("MAX_CONCURRENCY", "1")))
    parser.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "100")))

    parser.add_argument("--warmup-num-prompts", type=int, default=int(os.environ.get("WARMUP_NUM_PROMPTS", "5")))
    parser.add_argument("--warmup-input-len", type=int, default=int(os.environ.get("WARMUP_INPUT_LEN", "128")))
    parser.add_argument("--warmup-output-len", type=int, default=int(os.environ.get("WARMUP_OUTPUT_LEN", "32")))

    parser.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "/models"))
    parser.add_argument("--hf-token", default=os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

    parser.add_argument("--serve-host", default=os.environ.get("SERVE_HOST", "0.0.0.0"))
    parser.add_argument("--serve-port", type=int, default=int(os.environ.get("SERVE_PORT", "8000")))

    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "bfloat16"))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "8128")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95")),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--enforce-eager", action="store_true", default=os.environ.get("ENFORCE_EAGER", "1") != "0")
    parser.add_argument("--trust-remote-code", action="store_true", default=os.environ.get("TRUST_REMOTE_CODE", "1") != "0")
    parser.add_argument("--extra-vllm-args", default=os.environ.get("VLLM_EXTRA_ARGS", ""))

    parser.add_argument("--startup-timeout", type=int, default=int(os.environ.get("STARTUP_TIMEOUT", "1800")))
    parser.add_argument("--cooldown-seconds", type=int, default=int(os.environ.get("COOLDOWN_SECONDS", "5")))
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

    models = parse_csv(args.models)
    if not models:
        print("No models provided.")
        return 1

    length_pairs = parse_length_pairs(args.length_pairs)
    if not length_pairs:
        print("No length pairs provided.")
        return 1

    base_url = f"http://127.0.0.1:{args.serve_port}"
    health_url = f"{base_url}/health"

    results = []
    failed_models = []

    for model in models:
        safe_name = sanitize_model_name(model)
        local_dir = os.path.join(args.models_dir, safe_name)

        print("=" * 80)
        print(f"Model: {model}")
        print(f"Local dir: {local_dir}")
        print("=" * 80)

        try:
            download_model(model, local_dir, args.hf_token)
        except Exception as exc:
            print(f"Download failed for {model}: {exc}")
            failed_models.append((model, "download_failed"))
            if args.clear_model_after:
                shutil.rmtree(local_dir, ignore_errors=True)
            continue

        vllm_cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            args.serve_host,
            "--port",
            str(args.serve_port),
            "--model",
            local_dir,
            "--served-model-name",
            model,
            "--dtype",
            args.dtype,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
        ]
        if args.api_key:
            vllm_cmd.extend(["--api-key", args.api_key])
        if args.enforce_eager:
            vllm_cmd.append("--enforce-eager")
        if args.trust_remote_code:
            vllm_cmd.append("--trust-remote-code")
        if args.extra_vllm_args:
            vllm_cmd.extend(shlex.split(args.extra_vllm_args))

        print("Starting vLLM server:")
        print(" ".join(vllm_cmd))
        proc = subprocess.Popen(vllm_cmd, env=base_env)

        healthy = wait_for_health(proc, health_url, args.startup_timeout)
        if not healthy:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            failed_models.append((model, "vllm_start_failed"))
            if args.clear_model_after:
                shutil.rmtree(local_dir, ignore_errors=True)
            continue

        if args.warmup_num_prompts > 0:
            print("Running warmup...")
            run_bench(
                model_name=model,
                base_url=base_url,
                input_len=args.warmup_input_len,
                output_len=args.warmup_output_len,
                max_concurrency=min(args.max_concurrency, 4),
                num_prompts=args.warmup_num_prompts,
                api_key=args.api_key,
                env=base_env,
            )

        for input_len, output_len in length_pairs:
            print("-" * 80)
            print(f"Benchmarking {model} @ in={input_len}, out={output_len}")
            ok, output = run_bench(
                model_name=model,
                base_url=base_url,
                input_len=input_len,
                output_len=output_len,
                max_concurrency=args.max_concurrency,
                num_prompts=args.num_prompts,
                api_key=args.api_key,
                env=base_env,
            )
            results.append({
                "model": model,
                "input_len": input_len,
                "output_len": output_len,
                "success": ok,
            })
            if not ok:
                failed_models.append((model, f"bench_failed_{input_len}x{output_len}"))

        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

        if args.clear_model_after:
            shutil.rmtree(local_dir, ignore_errors=True)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        for item in results:
            status = "OK" if item["success"] else "FAIL"
            print(f"{status} | {item['model']} | in={item['input_len']} out={item['output_len']}")

    if failed_models:
        print("=" * 80)
        print("FAILED MODELS / RUNS")
        print("=" * 80)
        for model, reason in failed_models:
            print(f"{model}: {reason}")

    if failed_models and args.fail_on_error:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
