from dataclasses import dataclass
import torch
from contextvars import ContextVar

@dataclass
class CacheContext:
    """
    global context for kv cache during forward pass
    """
    layer_idx: int
    is_prefill: bool
    slot_mapping: torch.Tensor | None = None #  tells you exactly which physical positions in the flattened KV cache buffer to write the newly computed key/value vectors for the tokens being processed in the current forward pass.
    block_table: torch.Tensor | None = None
    context_len: int | None = None
    block_size: int = 16

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