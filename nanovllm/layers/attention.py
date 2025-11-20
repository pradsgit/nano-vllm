import torch
import torch.nn as nn
import torch.nn.functional as F
from nanovllm.utils.cache_context import get_context, CacheContext
from flash_attn import flash_attn_varlen_func

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

    def _write_to_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: CacheContext,
    ):
        """
        write k, v into kv_blocks at current layer's slice
        """
        slot_mapping = ctx.slot_mapping # (batch_sz, max_len)

        # vectorized write - no loop needed
        # self.k_cache and self.v_cache are just views of self.kv_cache storage object created in model_runner
        # so we are essentially writing to storage of self.kv_cache
        self.k_cache[slot_mapping] = k
        self.v_cache[slot_mapping] = v

    def _write_to_cache_batch(
        self,
        k: torch.Tensor, # (batch_sz, max_len, num_heads, head_dim)
        v: torch.Tensor,
        ctx: CacheContext
    ):
        slot_mapping = ctx.slot_mapping # (batch_sz, max_len)
        if ctx.is_prefill:
            valid_mask = slot_mapping >= 0
            valid_slots = slot_mapping[valid_mask] # flattens to 1D with valid slot position numbers(num_valid, )
            valid_k = k[valid_mask] # get valid tokens values leaving pad tokens values
            valid_v = v[valid_mask] # (num_valid, num_heads, dim)

            self.k_cache[valid_slots] = valid_k
            self.v_cache[valid_slots] = valid_v
        else:
            slot_mapping = slot_mapping.flatten() # (batch_sz, )
            k_squeezed = k.squeeze(1) # (batch_sz, num_heads, head_dim)
            v_squeezed = v.squeeze(1)

            self.k_cache[slot_mapping] = k_squeezed
            self.v_cache[slot_mapping] = v_squeezed

    def _read_from_cache_batch(self, ctx: CacheContext):
        """
        read past kv cache for the slots

        since slot_mapping resets for every model forward pass, we rebuild it on demand
        """
        block_tables = ctx.block_tables
        context_lens = ctx.context_lens
        block_size = ctx.block_size

        read_slot_mapping = self._build_full_slot_mapping_batch(
            block_tables,
            context_lens,
            block_size
        )
        k_past = self.k_cache[read_slot_mapping] # flattened (num_slots, num_heads, head_dim)
        v_past = self.v_cache[read_slot_mapping]

        # split each seq k_past and v_past using context_lens
        cached_lens = context_lens.tolist() # context_lens include currently processing token which is not saved in cache yet
        k_past_list = torch.split(k_past, cached_lens, dim=0)
        v_past_list = torch.split(v_past, cached_lens, dim=0)

        max_cached_len = max(cached_lens)

        # pad all k_cache and v_cache to max_cached_len
        padded_k_list = []
        padded_v_list = []

        for k_seq, seq_len in zip(k_past_list, cached_lens):
            pad_len = max_cached_len - seq_len
            if pad_len > 0:
                # Pad with zeros (or any value, will be masked)
                padding = torch.zeros(pad_len, self.num_kv_heads, self.head_dim, device=k_seq.device, dtype=k_seq.dtype)
                k_padded = torch.cat([padding, k_seq], dim=0)  # Left pad
            else:
                k_padded = k_seq
            padded_k_list.append(k_padded)
        
        for v_seq, seq_len in zip(v_past_list, cached_lens):
            pad_len = max_cached_len - seq_len
            if pad_len > 0:
                # Pad with zeros (or any value, will be masked)
                padding = torch.zeros(pad_len, self.num_kv_heads, self.head_dim, device=v_seq.device, dtype=v_seq.dtype)
                v_padded = torch.cat([padding, v_seq], dim=0)  # Left pad
            else:
                v_padded = v_seq
            padded_v_list.append(v_padded)

        # return (batch_size, max_cached_len, num_heads, head_dim)
        return (
            torch.stack(padded_k_list, dim=0),
            torch.stack(padded_v_list, dim=0)
        )

    def _build_full_slot_mapping_batch(
        self,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int
    ): 
        slot_mapping = []

        # build previously saved slots for each sequence

        for block_table, context_len in zip(block_tables.tolist(), context_lens.tolist()):
            # saved_kv_slots = context_len - 1 # context_len includes the current processing token which is not saved yet
            for logical_pos in range(context_len):
                block_idx = logical_pos // block_size
                pos_in_block = logical_pos % block_size
                # if ctx.layer_idx == 0:
                # print(block_table)
                # print(block_idx)
                physical_block_id = block_table[block_idx]
                slot = physical_block_id * block_size + pos_in_block
                slot_mapping.append(slot)

        return slot_mapping


    def _read_from_cache(self, ctx: CacheContext):
        """
        Read full context K/V from cache by building slot mapping on demand.

        This avoids storing huge flat mappings for long contexts.
        Instead, we rebuild the mapping from block_table + context_len each time.
        """
        # Get sequence metadata from context
        block_table = ctx.block_table  # List of block IDs for this sequence
        context_len = ctx.context_len # Total tokens saved in kv_cache (prompt + outputs)
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

    def store_kvcache(self, k, v, slot_mapping, block_size):
        # k, v: (num_tokens, num_heads, head_dim)
        # slot_mapping: (num_tokens,) with flat indices
        
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size
        
        # Index into 4D cache
        self.k_cache[block_indices, block_offsets] = k
        self.v_cache[block_indices, block_offsets] = v

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        ctx = get_context()
        
        # store new k,v into cache if initialized
        if self.k_cache.numel() > 0 and self.v_cache.numel() > 0:
            self.store_kvcache(k, v, ctx.slot_mapping, ctx.block_size)
            # Use cache for attention
            k_attn = self.k_cache
            v_attn = self.v_cache
        else:
            # cache not initialized, use raw k,v
            k_attn = k
            v_attn = v

        # Compute attention
        o = flash_attn_varlen_func(
            q,
            k_attn,
            v_attn,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            softmax_scale=self.attn_scale,
            causal=True,
            block_table=ctx.block_tables
        )
        
        return o

    def forward_old(self, q, k, v, causal_mask=None):
        ctx = get_context()

        if ctx.is_prefill:
            # Prefill: can write then read (all tokens are new)
            self._write_to_cache_batch(k, v, ctx)
            k_for_attn, v_for_attn = k, v

        else:
            k_past, v_past = self._read_from_cache_batch(ctx)
            k_for_attn = torch.cat([k_past, k], dim=1)
            v_for_attn = torch.cat([v_past, v], dim=1)
            self._write_to_cache_batch(k, v, ctx)

        output = self._scaled_dot_product_attention_batch(q, k_for_attn, v_for_attn, ctx.attention_mask, causal_mask, ctx)
        return output

    def _scaled_dot_product_attention_batch(
        self,
        q: torch.Tensor,  # (batch_sz, num_query_tokens, num_heads, head_dim)
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # (batch_sz, num_key_tokens) : 1=real, 0=pad
        causal_mask: torch.Tensor | None = None,
        ctx: CacheContext | None = None,
    ):

        batch_sz = q.size(0)
        num_query_tokens = q.size(1)
        num_key_tokens = k.size(1)

        # handle GQA: repeat K and V to match number of Q heads
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        q = q.transpose(1, 2)  # (batch_sz, num_heads, num_query_tokens, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.attn_scale

        # Create causal mask if not provided
        if causal_mask is None:
            if num_query_tokens > 1:
                # Prefill: create lower triangular causal mask
                causal_mask = torch.tril(
                    torch.ones(num_query_tokens, num_key_tokens, dtype=torch.bool, device=q.device)
                )  # (num_query_tokens, num_key_tokens)
            else:
                # Decode: query attends to all keys
                causal_mask = torch.ones(
                    num_query_tokens, num_key_tokens, dtype=torch.bool, device=q.device
                )
        # Reshape for broadcasting: (1, 1, num_query_tokens, num_key_tokens)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch_sz, num_key_tokens) - which key positions are valid
            # Reshape for broadcasting: (batch_sz, 1, 1, num_key_tokens)
            padding_mask = attention_mask.view(batch_sz, 1, 1, num_key_tokens)

            # Combine: position (i, j) is valid if both causal AND not padding
            combined_mask = combined_mask & padding_mask  # Broadcast to (batch_sz, 1, num_query_tokens, num_key_tokens)

        # Apply combined mask
        attn_scores = attn_scores.masked_fill(~combined_mask, float('-inf'))

        # FIX: Replace rows with all -inf with 0 to prevent NaN
        all_neginf = torch.all(torch.isinf(attn_scores) & (attn_scores < 0), dim=-1, keepdim=True)
        attn_scores = torch.where(all_neginf, torch.zeros_like(attn_scores), attn_scores)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # attn_weights @ V
        output = torch.matmul(attn_weights, v)  # (batch_sz, num_heads, num_query_tokens, head_dim)
        output = output.transpose(1, 2)  # (batch_sz, num_query_tokens, num_heads, head_dim)

        return output

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor, # (batch_sz, num_query_tokens, num_heads, head_dim)
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
        # (batch_sz, num_heads, num_tokens, head_dim) @ (batch_sz, num_heads, head_dim, num_tokens)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.attn_scale

        # attn_scores shape is (batcjh_sz, num_heads, num_tokens, num_tokens)

        # apply casual mask, TODO: cache this in __init__ to avoid creating this on every forward pass

        if causal_mask is None:
            mask = torch.ones(num_query_tokens, num_key_tokens, dtype=torch.bool, device=q.device)

            if num_query_tokens > 1:
                # this is prefill phase
                mask = torch.tril(mask)

            # During decode (num_query_tokens == 1): mask is all True cause query attends to all the keys including current one

            causal_mask = mask.unsqueeze_(0)

        attn_scores.masked_fill_(~causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v) # (num_heads, num_tokens, num_tokens) @ (num_heads, num_tokens, head_dim) => (num_heads, num_tokens, head_dim)

        return output.transpose(0, 1) # Back to (num_tokens, num_heads, head_dim) non-contiguous output