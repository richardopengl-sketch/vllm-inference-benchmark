import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
import yaml

@dataclass
class RequestResult:
    concurrency: int
    request_id: int
    success: bool
    latency_sec: float
    output_tokens: int
    ttft_sec: Optional[float] = None
    tpot_sec: Optional[float] = None
    output_chunks: int = 0
    error: str = ""


def calculate_streaming_metrics(
    start_time: float,
    end_time: float,
    first_token_time: Optional[float],
    last_token_time: Optional[float],
    output_tokens: int,
) -> tuple[float, Optional[float], Optional[float]]:
    latency_sec = end_time - start_time
    ttft_sec = (
        first_token_time - start_time if first_token_time is not None else None
    )
    tpot_sec = None
    if (
        output_tokens > 1
        and first_token_time is not None
        and last_token_time is not None
    ):
        tpot_sec = (last_token_time - first_token_time) / (output_tokens - 1)

    return latency_sec, ttft_sec, tpot_sec

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
    stream: bool = False,
) -> RequestResult:
    url = f"{api_base}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    start = time.perf_counter()

    try:
        async with session.post(url, json=payload, timeout=300) as resp:
            if resp.status != 200:
                text = await resp.text()
                latency = time.perf_counter() - start
                return RequestResult(
                    concurrency=concurrency,
                    request_id=request_id,
                    success=False,
                    latency_sec=latency,
                    output_tokens=0,
                    error=f"HTTP {resp.status}: {text[:300]}",
                )

            if stream:
                first_token_time = None
                last_token_time = None
                output_tokens = 0
                output_chunks = 0
                received_done = False

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    event_data = line[len("data:") :].strip()
                    if event_data == "[DONE]":
                        received_done = True
                        break

                    event = json.loads(event_data)
                    usage = event.get("usage")
                    if usage is not None:
                        output_tokens = int(usage.get("completion_tokens", 0))

                    choices = event.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content")
                        if content:
                            token_time = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = token_time
                            last_token_time = token_time
                            output_chunks += 1

                end = time.perf_counter()
                latency, ttft, tpot = calculate_streaming_metrics(
                    start,
                    end,
                    first_token_time,
                    last_token_time,
                    output_tokens,
                )
                if not received_done:
                    return RequestResult(
                        concurrency=concurrency,
                        request_id=request_id,
                        success=False,
                        latency_sec=latency,
                        output_tokens=output_tokens,
                        ttft_sec=ttft,
                        tpot_sec=tpot,
                        output_chunks=output_chunks,
                        error="Streaming response ended before [DONE]",
                    )

                return RequestResult(
                    concurrency=concurrency,
                    request_id=request_id,
                    success=True,
                    latency_sec=latency,
                    output_tokens=output_tokens,
                    ttft_sec=ttft,
                    tpot_sec=tpot,
                    output_chunks=output_chunks,
                )

            latency = time.perf_counter() - start
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
    stream = bool(config.get("stream", True))

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
                    stream=stream,
                )
                results.append(result)

        tasks = [asyncio.create_task(worker(i)) for i in range(num_requests)]
        await asyncio.gather(*tasks)

    return results

def percentile_nearest_rank(values: List[float], percentile: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    rank = math.ceil(percentile * len(sorted_values))
    index = min(len(sorted_values) - 1, max(0, rank - 1))
    return sorted_values[index]

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
            "avg_ttft": None,
            "p50_ttft": None,
            "p95_ttft": None,
            "avg_tpot": None,
            "p50_tpot": None,
            "p95_tpot": None,
            "total_output_tokens": 0,
        }

    latencies = sorted(r.latency_sec for r in ok)
    ttfts = sorted(r.ttft_sec for r in ok if r.ttft_sec is not None)
    tpots = sorted(r.tpot_sec for r in ok if r.tpot_sec is not None)
    total_output_tokens = sum(r.output_tokens for r in ok)

    def aggregate(values: List[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if not values:
            return None, None, None
        p95_index = percentile_nearest_rank(values, 0.95)
        return (
            statistics.mean(values),
            statistics.median(values),
            p95_index,
        )

    avg_latency, p50_latency, p95_latency = aggregate(latencies)
    avg_ttft, p50_ttft, p95_ttft = aggregate(ttfts)
    avg_tpot, p50_tpot, p95_tpot = aggregate(tpots)

    return {
        "success": len(ok),
        "failed": len(failed),
        "avg_latency": avg_latency,
        "p50_latency": p50_latency,
        "p95_latency": p95_latency,
        "avg_ttft": avg_ttft,
        "p50_ttft": p50_ttft,
        "p95_ttft": p95_ttft,
        "avg_tpot": avg_tpot,
        "p50_tpot": p50_tpot,
        "p95_tpot": p95_tpot,
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
                "ttft_sec",
                "tpot_sec",
                "output_chunks",
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
        print(f"Avg TTFT: {summary['avg_ttft']}")
        print(f"P50 TTFT: {summary['p50_ttft']}")
        print(f"P95 TTFT: {summary['p95_ttft']}")
        print(f"Avg TPOT: {summary['avg_tpot']}")
        print(f"P50 TPOT: {summary['p50_tpot']}")
        print(f"P95 TPOT: {summary['p95_tpot']}")
        print(f"Total output tokens: {summary['total_output_tokens']}")

        all_results.extend(results)

    write_results_csv(args.output, all_results)
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
