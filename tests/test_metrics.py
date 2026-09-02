import pytest

from benchmark.benchmark_client import (
    RequestResult,
    calculate_streaming_metrics,
    summarize,
)


def test_summarize_success_results():
    results = [
        RequestResult(
            concurrency=1,
            request_id=0,
            success=True,
            latency_sec=1.0,
            output_tokens=10,
            ttft_sec=0.1,
            tpot_sec=0.02,
        ),
        RequestResult(
            concurrency=1,
            request_id=1,
            success=True,
            latency_sec=2.0,
            output_tokens=20,
            ttft_sec=0.2,
            tpot_sec=0.03,
        ),
        RequestResult(
            concurrency=1,
            request_id=2,
            success=True,
            latency_sec=3.0,
            output_tokens=30,
            ttft_sec=0.3,
            tpot_sec=0.04,
        ),
    ]

    summary = summarize(results)

    assert summary["success"] == 3
    assert summary["failed"] == 0

    assert summary["avg_latency"] == pytest.approx(2.0)
    assert summary["p50_latency"] == pytest.approx(2.0)
    assert summary["p95_latency"] == pytest.approx(3.0)

    assert summary["avg_ttft"] == pytest.approx(0.2)
    assert summary["p50_ttft"] == pytest.approx(0.2)
    assert summary["p95_ttft"] == pytest.approx(0.3)

    assert summary["avg_tpot"] == pytest.approx(0.03)
    assert summary["p50_tpot"] == pytest.approx(0.03)
    assert summary["p95_tpot"] == pytest.approx(0.04)

    assert summary["total_output_tokens"] == 60


def test_summarize_failed_results():
    results = [
        RequestResult(
            concurrency=1,
            request_id=0,
            success=False,
            latency_sec=0.5,
            output_tokens=0,
            error="connection refused",
        ),
        RequestResult(
            concurrency=1,
            request_id=1,
            success=False,
            latency_sec=0.6,
            output_tokens=0,
            error="connection refused",
        ),
    ]

    summary = summarize(results)

    assert summary["success"] == 0
    assert summary["failed"] == 2

    assert summary["avg_latency"] is None
    assert summary["p50_latency"] is None
    assert summary["p95_latency"] is None

    assert summary["avg_ttft"] is None
    assert summary["p50_ttft"] is None
    assert summary["p95_ttft"] is None

    assert summary["avg_tpot"] is None
    assert summary["p50_tpot"] is None
    assert summary["p95_tpot"] is None

    assert summary["total_output_tokens"] == 0

def test_summarize_mixed_results():
    results = [
        RequestResult(
            concurrency=2,
            request_id=0,
            success=True,
            latency_sec=1.0,
            output_tokens=8,
            ttft_sec=0.25,
            tpot_sec=0.05,
        ),
        RequestResult(
            concurrency=2,
            request_id=1,
            success=False,
            latency_sec=2.0,
            output_tokens=0,
            error="timeout",
        ),
    ]

    summary = summarize(results)

    assert summary["success"] == 1
    assert summary["failed"] == 1
    assert summary["avg_latency"] == 1.0
    assert summary["p50_latency"] == 1.0
    assert summary["avg_ttft"] == pytest.approx(0.25)
    assert summary["p50_ttft"] == pytest.approx(0.25)
    assert summary["p95_ttft"] == pytest.approx(0.25)
    assert summary["avg_tpot"] == pytest.approx(0.05)
    assert summary["p50_tpot"] == pytest.approx(0.05)
    assert summary["p95_tpot"] == pytest.approx(0.05)
    assert summary["total_output_tokens"] == 8


def test_calculate_streaming_metrics_multi_token_stream():
    latency, ttft, tpot = calculate_streaming_metrics(
        start_time=10.0,
        end_time=12.0,
        first_token_time=10.2,
        last_token_time=11.1,
        output_tokens=4,
    )

    assert latency == pytest.approx(2.0)
    assert ttft == pytest.approx(0.2)
    assert tpot == pytest.approx(0.3)


def test_calculate_streaming_metrics_one_token_stream():
    latency, ttft, tpot = calculate_streaming_metrics(
        start_time=10.0,
        end_time=10.5,
        first_token_time=10.2,
        last_token_time=10.2,
        output_tokens=1,
    )

    assert latency == pytest.approx(0.5)
    assert ttft == pytest.approx(0.2)
    assert tpot is None


def test_calculate_streaming_metrics_without_first_token():
    latency, ttft, tpot = calculate_streaming_metrics(
        start_time=10.0,
        end_time=10.5,
        first_token_time=None,
        last_token_time=None,
        output_tokens=0,
    )

    assert latency == pytest.approx(0.5)
    assert ttft is None
    assert tpot is None
