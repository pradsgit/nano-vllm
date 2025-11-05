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
        self.num_cached_tokens = 0

    def _init_cache(
        self, 
        device: torch.device | None,
        dtype: torch.dtype | None,
    ): 
        self.k_cache = torch.zeros(
            self.max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

        self.v_cache = torch.zeros(
            self.max_seq_len,
            self.num_kv_heads,
            self.head_dim,
            device=device,
            dtype=dtype
        )

        self.num_cached_tokens = 0

    def _write_to_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: CacheContext,
        num_tokens: int,
    ):
        """
        write k, v into kv_blocks at current layer's slice
        """
        kv_blocks = ctx.kv_blocks # [tensor, tensor] each tensor shape: (2, num_hidden_layers, block_size, num_kv_heads, head_dim)
        layer_idx = ctx.layer_idx
        slot_idx = ctx.slot_idx
        block_size = kv_blocks[0].size(2)

        for i in range(num_tokens):
            pos = slot_idx + i
            block_num = pos // block_size
            pos_in_block = pos % block_size

            kv_blocks[block_num][0, layer_idx, pos_in_block] = k[i]
            kv_blocks[block_num][1, layer_idx, pos_in_block] = v[i]

    def _read_from_cache(self, ctx: CacheContext, total_tokens: int):
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

        print(f'read cache from kv_blocks')
        print(f'k shape is {k_all.shape}')
        print(f'v shape is {v_all.shape}')
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
        self._write_to_cache(k, v, ctx, num_new_tokens)
        
        # initialize cache on first use
        # if self.k_cache is None:
        #     # this is prefill step
        #     self._init_cache(k.device, k.dtype)

        if ctx.is_prefill:
            k_for_attn, v_for_attn = k, v
        else:
            k_for_attn, v_for_attn = self._read_from_cache(ctx, ctx.slot_idx + num_new_tokens)

        # if is_prefill:
        #     k_for_attn, v_for_attn = self._prefill(k, v)
        # else:
        #     k_for_attn, v_for_attn = self._decode(k, v)

        output = self._scaled_dot_product_attention(q, k_for_attn, v_for_attn, causal_mask)

        return output
    
    def _prefill(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        prefill: Sstore all prompt tokens in cache.
        
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