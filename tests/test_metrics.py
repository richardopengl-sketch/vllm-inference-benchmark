from benchmark.benchmark_client import RequestResult, summarize


def test_summarize_success_results():
    results = [
        RequestResult(
            concurrency=1,
            request_id=0,
            success=True,
            latency_sec=1.0,
            output_tokens=10,
        ),
        RequestResult(
            concurrency=1,
            request_id=1,
            success=True,
            latency_sec=2.0,
            output_tokens=20,
        ),
        RequestResult(
            concurrency=1,
            request_id=2,
            success=True,
            latency_sec=3.0,
            output_tokens=30,
        ),
    ]

    summary = summarize(results)

    assert summary["success"] == 3
    assert summary["failed"] == 0
    assert summary["avg_latency"] == 2.0
    assert summary["p50_latency"] == 2.0
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
    assert summary["total_output_tokens"] == 0


def test_summarize_mixed_results():
    results = [
        RequestResult(
            concurrency=2,
            request_id=0,
            success=True,
            latency_sec=1.0,
            output_tokens=8,
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
    assert summary["total_output_tokens"] == 8
