# vLLM Request Lifecycle Notes

## High-level flow

User request
→ OpenAI-compatible API server
→ AsyncLLMEngine
→ LLMEngine
→ Scheduler
→ KV cache / block manager
→ ModelRunner / Executor
→ GPU kernels

## Questions to answer

- Where does the HTTP request enter vLLM?
- How does vLLM enqueue a generation request?
- Where does scheduling happen?
- How does vLLM represent KV cache blocks?
- Where are prefill and decode separated?

## Terms

### Prefill

The phase that processes the input prompt and builds the initial KV cache.

### Decode

The token-by-token generation phase after prefill.

### KV cache

Cached key/value tensors from attention layers, reused during decoding.

### Continuous batching

A serving technique that mixes requests at different generation stages to keep the GPU busy.