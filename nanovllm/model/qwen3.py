import torch
import torch.nn as nn
from nanovllm.layers.rope import get_rope
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.attention import Attention

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int, # number of q heads
        num_kv_heads: int, # number of kv heads
        head_dim: int | None = None,
        qkv_bias: bool = False,
        max_position: int = 4096 * 32,
        rms_norm_eps: float = 1e-6,
        rope_theta: int=10000,
        rope_scaling: tuple | None = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // self.num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=qkv_bias)
        self.kv_proj = nn.Linear(hidden_size, 2 * self.kv_size, bias=qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rope = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position = max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)

        self.attn_scale = self.head_dim ** -0.5
        self.attention = Attention(num_heads, self.head_dim, num_kv_heads, self.attn_scale, max_position)

    def forward(
        self,
        x: torch.Tensor, # (num_tokens, hidden_size)
        positions: torch.Tensor, # (num_tokens,) - position indices
    ): 
        # shape of x is (seq_len, hidden_size) phase1: considers only one sequence
        seq_len, hidden_size = x.size()
        assert self.hidden_size == hidden_size

        # project x to qkv
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        kv = self.kv_proj(x).view(-1, 2, self.num_kv_heads, self.head_dim).permute(1, 0, 2, 3)
        k, v = kv.unbind(0)

        # apply qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # apply rope to q and k
        q, k = self.rope(positions, q, k)

        # apply Grouped Query Attention
        # attention with KV caching
        # Input: (num_tokens, num_heads, head_dim)
        # Output: (num_tokens, num_heads, head_dim)
        attn_output = self.attention(q, k, v)

        # flatten heads back
        # (num_tokens, num_heads, head_dim) -> (num_tokens, num_heads * head_dim)
        attn_output = attn_output.reshape(-1, self.q_size)

        output = self.o_proj(attn_output)

        return output