from dataclasses import dataclass
import torch
from contextvars import ContextVar

@dataclass
class CacheContext:
    """
    global context for kv cache during forward pass
    """
    kv_blocks: list[torch.Tensor]  # blocks from block manager
    slot_idx: int
    layer_idx: int
    is_prefill: bool

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