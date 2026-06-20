import argparse
import asyncio
import csv
import statistics
import time
from dataclasses import dataclass
from typing import List

import aiohttp
import yaml


@dataclass
class RequestResult:
    concurrency: int
    request_id: int
    success: bool
    latency_sec: float
    output_tokens: int
    error: str = ""


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def send_one_request(
    session: aiohttp.ClientSession,
    api_base: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    request_id: int,
) -> RequestResult:
    url = f"{api_base}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    start = time.perf_counter()

    try:
        async with session.post(url, json=payload, timeout=300) as resp:
            text = await resp.text()
            latency = time.perf_counter() - start

            if resp.status != 200:
                return RequestResult(
                    concurrency=concurrency,
                    request_id=request_id,
                    success=False,
                    latency_sec=latency,
                    output_tokens=0,
                    error=f"HTTP {resp.status}: {text[:300]}",
                )

            data = await resp.json()
            usage = data.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)

            return RequestResult(
                concurrency=concurrency,
                request_id=request_id,
                success=True,
                latency_sec=latency,
                output_tokens=output_tokens,
            )

    except Exception as e:
        latency = time.perf_counter() - start
        return RequestResult(
            concurrency=concurrency,
            request_id=request_id,
            success=False,
            latency_sec=latency,
            output_tokens=0,
            error=repr(e),
        )


async def run_concurrency_level(config: dict, concurrency: int) -> List[RequestResult]:
    api_base = config["api_base"]
    model = config["model"]
    prompts = config["prompts"]
    num_requests = int(config["num_requests"])
    max_tokens = int(config["max_tokens"])
    temperature = float(config["temperature"])

    semaphore = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []

    async with aiohttp.ClientSession() as session:

        async def worker(request_id: int):
            async with semaphore:
                prompt = prompts[request_id % len(prompts)]
                result = await send_one_request(
                    session=session,
                    api_base=api_base,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    concurrency=concurrency,
                    request_id=request_id,
                )
                results.append(result)

        tasks = [asyncio.create_task(worker(i)) for i in range(num_requests)]
        await asyncio.gather(*tasks)

    return results


def summarize(results: List[RequestResult]) -> dict:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not ok:
        return {
            "success": 0,
            "failed": len(failed),
            "avg_latency": None,
            "p50_latency": None,
            "p95_latency": None,
            "total_output_tokens": 0,
        }

    latencies = sorted(r.latency_sec for r in ok)
    total_output_tokens = sum(r.output_tokens for r in ok)

    p50 = statistics.median(latencies)
    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    p95 = latencies[p95_index]

    return {
        "success": len(ok),
        "failed": len(failed),
        "avg_latency": statistics.mean(latencies),
        "p50_latency": p50,
        "p95_latency": p95,
        "total_output_tokens": total_output_tokens,
    }


async def check_server_ready(api_base: str) -> bool:
    url = f"{api_base}/models"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return True

                text = await resp.text()
                print(f"Server health check failed. HTTP {resp.status}: {text[:300]}")
                return False

    except Exception as e:
        print(f"Server is not reachable at {api_base}. Error: {repr(e)}")
        return False


def write_results_csv(path: str, results: List[RequestResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concurrency",
                "request_id",
                "success",
                "latency_sec",
                "output_tokens",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen_0_5b.yaml")
    parser.add_argument("--output", default="results/baseline.csv")
    args = parser.parse_args()

    config = load_config(args.config)

    server_ready = await check_server_ready(config["api_base"])
    if not server_ready:
        print("")
        print("vLLM server is not running or not reachable.")
        print("Later on Azure GPU VM, start it with:")
        print("  bash scripts/start_vllm_server.sh")
        return

    all_results: List[RequestResult] = []

    for concurrency in config["concurrency_levels"]:
        print(f"\nRunning concurrency={concurrency}")
        start = time.perf_counter()

        results = await run_concurrency_level(config, int(concurrency))
        elapsed = time.perf_counter() - start
        summary = summarize(results)

        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Success: {summary['success']}, Failed: {summary['failed']}")
        print(f"Avg latency: {summary['avg_latency']}")
        print(f"P50 latency: {summary['p50_latency']}")
        print(f"P95 latency: {summary['p95_latency']}")
        print(f"Total output tokens: {summary['total_output_tokens']}")

        all_results.extend(results)

    write_results_csv(args.output, all_results)
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
