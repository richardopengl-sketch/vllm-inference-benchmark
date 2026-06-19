# vLLM Inference Benchmark

A learning and benchmarking project for LLM inference serving systems.

## Goals

- Benchmark vLLM serving performance under concurrent workloads.
- Measure TTFT, TPOT, throughput, p50/p95 latency, and GPU memory usage.
- Understand prefill/decode bottlenecks, KV cache growth, and continuous batching behavior.
- Compare future serving frameworks such as vLLM and SGLang.

## Planned Experiments

- Qwen2.5-0.5B baseline on Azure T4
- Concurrency sweep: 1, 2, 4, 8, 16
- Input length sweep: 128, 512, 2048
- Output length sweep: 64, 256, 1024