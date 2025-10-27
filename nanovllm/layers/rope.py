import torch
import torch.nn as nn
from functools import lru_cache

def apply_rope_emb(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1).flatten(-2)

class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_seq_len: int,
        base: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        self.rotary_dim = rotary_dim
        assert rotary_dim % 2 == 0, "dimension should be even in RotaryPositionEmbedding"

        inv_freqs = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", positions, inv_freqs) # freqs shape is [max_seq_len, rotary_dim // 2]

        # cache cos and sin values of the frequencies
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1) # shape is (max_seq_len, 1, rotary_dim)
        self.register_buffer('cos_sin_cache', cache, persistent=False)

    @torch.compile
    def forward(
        self,
        token_positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        cos_sin = self.cos_sin_cache[token_positions]
        cos, sin = cos_sin.chunk(2, dim=-1) # each shape is (max_seq_len, 1, rotary_dim // 2)
        query = apply_rope_emb(query, cos, sin)
        key = apply_rope_emb(key, cos, sin)

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryPositionEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb