# nano-vLLM

A minimal implementation of vLLM's v1 architecture core inference optimizations, built from scratch to understand modern LLM serving.

## Why?

Modern inference engines achieve 2-4x higher throughput than naive serving through sophisticated memory management and batching strategies. This project implements the key techniques to understand how they actually work.

## Features

✅ **Continuous Batching** - Dynamic request scheduling with prefill/decode interleaving
✅ **PagedAttention** - Non-contiguous KV cache allocation with block-based memory management  
✅ **Flash Attention** - Memory-efficient attention computation  
✅ **Prefix Caching** - Hash-based block sharing for sequences with common prefixes
✅ **Chunked prefill** - Process long sequences in manageable chunks to reduce throttling of other requests

🚧 **Coming Soon**: CUDA graphs, tensor parallelism, quantization

## Key Insights

### Physical vs Logical Paging
The KV cache is a single contiguous GPU tensor, but sequences use non-contiguous blocks within it. "Paging" is purely logical - blocks are indexed, not scattered in physical memory.

### Memory Bandwidth is the Bottleneck
During decode, GPU utilization stays <40% even at large batch sizes because we're memory-bound, not compute-bound. Data movement dominates latency.

### Prefix Caching via Incremental Hashing
Full blocks are cached using xxHash of token IDs + previous block hash. Two sequences with identical prefixes automatically share KV cache blocks through ref counting.

## Architecture
```
Scheduler
  ├─> BlockManager (manages KV cache blocks)
  │     ├─> Free list (doubly linked list)
  │     ├─> Cache lookup (hash → block_id)
  │     └─> Ref counting (block sharing)
  │
  └─> Sequences (waiting → running → finished)

KV Cache: [2, num_layers, num_blocks, block_size, num_heads, head_dim]
           ↑                ↑           ↑
           k/v            logical     tokens per
                          blocks      block
```

## Usage
See `example.py` for usage

## Benchmarks

TODO: Add throughput comparisons vs naive batching

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py)

## License

MIT

---

*Built to learn. Not production-ready*