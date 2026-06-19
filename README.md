# vLLM Inference Benchmark

A learning and benchmarking project for LLM inference serving systems.

## Goals

- Benchmark vLLM serving performance under concurrent workloads.
- Measure TTFT, TPOT, throughput, p50/p95 latency, and GPU memory usage.
- Understand prefill/decode bottlenecks, KV cache growth, and continuous batching behavior.
- Compare future serving frameworks such as vLLM and SGLang.

## Initial Model

- Qwen/Qwen2.5-0.5B-Instruct

## Metrics

- End-to-end latency
- TTFT: Time To First Token
- TPOT: Time Per Output Token
- p50 / p95 latency
- Throughput: requests/sec
- Throughput: output tokens/sec
- GPU memory usage

## Week 1 Scope

- Build initial benchmark client.
- Add vLLM server readiness check.
- Prepare Azure GPU VM startup scripts.
- Read vLLM request lifecycle code path.

## Future Experiments

- Concurrency sweep: 1, 2, 4, 8, 16
- Input length sweep: 128, 512, 2048
- Output length sweep: 64, 256, 1024
- vLLM vs SGLang comparison
- T4 vs A10 vs A100 comparison