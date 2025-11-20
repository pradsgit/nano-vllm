from dataclasses import dataclass
import torch
from contextvars import ContextVar
from typing import Optional

@dataclass
class CacheContext:
    """
    global context for kv cache during forward pass
    """
    layer_idx: int = -1
    is_prefill: bool = False
    slot_mapping: torch.Tensor | None = None #  tells you exactly which physical positions in the flattened KV cache buffer to write the newly computed key/value vectors for the tokens being processed in the current forward pass.
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None

    attention_mask: torch.Tensor | None = None
    block_size: int = 256

_cache_context: ContextVar[CacheContext | None] = ContextVar('CacheContext', default=None)

def set_context(ctx: CacheContext) -> None:
    """
    set global cache context
    """
    _cache_context.set(ctx)

def get_context() -> CacheContext:
    """
    get global cache context
    """
    ctx = _cache_context.get()
    if ctx is None:
        raise RuntimeError("CacheContext not set. Call set_context() before forward pass.")
    return ctx

@dataclass
class ModelInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    attention_mask: Optional[torch.Tensor]


@dataclass
class GenerationMetrics:
    prefill_time: float = 0
    decode_times: list[float] = None
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0

    def __post_init__(self):
        if self.decode_times is None:
            self.decode_times = []

    def print_summary(self):
        total_time = self.prefill_time + sum(self.decode_times)
        avg_decode = sum(self.decode_times) / len(self.decode_times) if self.decode_times else 0
        throughput = self.num_generated_tokens / total_time if total_time > 0 else 0

        print(f"\n{'='*50}")
        print(f"Prompt tokens: {self.num_prompt_tokens}")
        print(f"Generated tokens: {self.num_generated_tokens}")
        print(f"TTFT (prefill): {self.prefill_time:.3f}s")
        print(f"Avg TPOT (decode): {avg_decode*1000:.2f}ms")
        print(f"Total time: {total_time:.3f}s")
        print(f"Throughput: {throughput:.2f} tok/s")
        print(f"{'='*50}\n")