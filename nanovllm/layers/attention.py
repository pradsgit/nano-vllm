import torch
import torch.nn as nn
import torch.nn.functional as F
from nanovllm.utils.cache_context import get_context, CacheContext

class Attention(nn.Module):
    """
    Basic attention implementation.
    
    Input shapes for Phase1 only single sequence handling:
    - q: (num_tokens, num_heads, head_dim)
    - k: (num_tokens, num_kv_heads, head_dim)
    - v: (num_tokens, num_kv_heads, head_dim)
    """
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        attn_scale: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_scale = attn_scale
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len

        # for gqa: how many q heads for kv heads
        assert num_heads % num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads

        # simple contiguous cache structures
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        # self.num_cached_tokens = 0

    def _write_to_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: CacheContext,
        # num_tokens: int,
    ):
        """
        write k, v into kv_blocks at current layer's slice
        """
        slot_mapping = ctx.slot_mapping # (num_tokens, )

        # vectorized write - no loop needed
        self.k_cache[slot_mapping] = k
        self.v_cache[slot_mapping] = v

    def _read_from_cache(self, ctx: CacheContext):
        """
        Read full context K/V from cache by building slot mapping on demand.
        
        This avoids storing huge flat mappings for long contexts.
        Instead, we rebuild the mapping from block_table + context_len each time.
        """
        # Get sequence metadata from context
        block_table = ctx.block_table  # List of block IDs for this sequence
        context_len = ctx.context_len  # Total tokens in context (prompt + outputs)
        block_size = ctx.block_size    # Tokens per block
        
        # Build full slot mapping for positions [0, context_len)
        read_slot_mapping = self._build_full_slot_mapping(
            block_table=block_table,
            context_len=context_len,
            block_size=block_size
        )
        
        # Vectorized read
        k_all = self.k_cache[read_slot_mapping]  # [context_len, num_kv_heads, head_dim]
        v_all = self.v_cache[read_slot_mapping]
        
        return k_all, v_all

    def _build_full_slot_mapping(
        self,
        block_table: list[int],
        context_len: int,
        block_size: int
    ) -> torch.Tensor:
        """
        Build slot mapping for logical positions [0, context_len) on demand.
        
        Example:
            block_table = [10, 11], context_len = 6, block_size = 4
            → slots = [40,41,42,43,44,45]
        """
        if context_len == 0:
            return torch.empty(0, dtype=torch.long, device='cuda')
        
        slots = []
        for logical_pos in range(context_len):
            block_idx = logical_pos // block_size
            pos_in_block = logical_pos % block_size
            physical_block_id = block_table[block_idx]
            slot = physical_block_id * block_size + pos_in_block
            slots.append(slot)
        
        return torch.tensor(slots, dtype=torch.long, device='cuda')


    def _read_from_cache_old(self, ctx: CacheContext, total_tokens: int):
        """
        read all k,v cached blocks for this current layer
        """
        kv_blocks = ctx.kv_blocks 
        # eg: kv_blocks is list of tensors eg: [tensor, tensor] 
        # each tensor in the list is of shape: (2, num_hidden_layers, block_size, num_kv_heads, head_dim)
        layer_idx = ctx.layer_idx
        block_size = kv_blocks[0].size(2) 

        k_list, v_list = [], []

        for i in range(total_tokens):
            block_num = i // block_size
            pos_in_block = i % block_size

            k_list.append(kv_blocks[block_num][0, layer_idx, pos_in_block])
            v_list.append(kv_blocks[block_num][1, layer_idx, pos_in_block])

        k_all = torch.stack(k_list, dim=0)
        v_all = torch.stack(v_list, dim=0)

        return k_all, v_all

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask = None,
    ):
        """
        Forward pass with "contiguous" KV caching.
        # cache K and V values during prefill step and use them during decode step
        
        Args:
            q, k, v: New tokens to process (could be 1 for decode, many for prefill)
        
        Returns:
            output: (num_tokens, num_heads, head_dim)
        """
        ctx = get_context()

        num_new_tokens = k.size(0)
        self._write_to_cache(k, v, ctx)

        if ctx.is_prefill:
            k_for_attn, v_for_attn = k, v
        else:
            k_for_attn, v_for_attn = self._read_from_cache(ctx)

        output = self._scaled_dot_product_attention(q, k_for_attn, v_for_attn, causal_mask)

        return output
    
    def _prefill(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        prefill: store all prompt tokens in cache.
        
        returns:
            k_for_attn, v_for_attn: same as input (all new tokens)
        """
        num_tokens = k.size(0)
        # initial filling of k_cache and v_cache
        self.k_cache[:num_tokens] = k
        self.v_cache[:num_tokens] = v
        self.num_cached_tokens = num_tokens

        return k, v
    
    def _decode(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        decode: Append new token to cache and return all cached.
        
        Returns:
            k_for_attn, v_for_attn: All cached tokens including new one
        """
        pos = self.num_cached_tokens
        self.k_cache[pos] = k.squeeze(0) # remove first dimension cause it is 1 in decode phase
        self.v_cache[pos] = v.squeeze(0)
        self.num_cached_tokens += 1

        # return all the tokens k and v
        k_all = self.k_cache[:self.num_cached_tokens] # k_all is a new tensor view that points to a portion of self.k_cache's storage
        v_all = self.v_cache[:self.num_cached_tokens]
        return k_all, v_all

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask=None,
    ):
        num_query_tokens = q.size(0)
        num_key_tokens = k.size(0)
    
        # handle GQA: repeat K and V to match number of Q heads
        if self.num_kv_heads != self.num_heads:
            # (num_tokens, num_kv_heads, head_dim) 
            # → (num_tokens, num_heads, head_dim)
            # TODO: check if dim=1 would be valid for batched sequences
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1) # dim value must be head dimension of k
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        q = q.transpose(0, 1) # (num_heads, num_tokens, head_dim)
        k = k.transpose(0, 1) # (num_heads, num_tokens, head_dim)
        v = v.transpose(0, 1) # (num_heads, num_tokens, head_dim)

        # compute attention scores: Q @ K^T
        # (num_heads, num_tokens, head_dim) @ (num_heads, head_dim, num_tokens)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.attn_scale

        # attn_scores shape is (num_heads, num_tokens, num_tokens)

        # apply casual mask, TODO: cache this in __init__ to avoid creating this on every forward pass

        if causal_mask is None:
            mask = torch.ones(num_query_tokens, num_key_tokens, dtype=torch.bool, device=q.device)

            if num_query_tokens > 1:
                # this is prefill phase
                mask = torch.tril(mask)

            # During decode (num_query_tokens == 1): mask is all True

            causal_mask = mask.unsqueeze_(0)

        attn_scores.masked_fill_(~causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v) # (num_heads, num_tokens, num_tokens) @ (num_heads, num_tokens, head_dim) => (num_heads, num_tokens, head_dim)

        return output.transpose(0, 1) # Back to (num_tokens, num_heads, head_dim) non-contiguous output