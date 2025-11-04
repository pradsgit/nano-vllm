import torch
import torch.nn as nn
from transformers import Qwen3Config

from nanovllm.layers.rope import get_rope
from nanovllm.layers.activation import SiluAndMul
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
        max_position: int = 4096 * 32, # whats the significance of this
        rms_norm_eps: float = 1e-6,
        rope_theta: int=10000,
        rope_scaling: tuple | None = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // self.num_heads

        # Never assume num_heads * head_dim = hidden_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=qkv_bias)

        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)

        self.rope = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position = max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.self_attn_scale = self.head_dim ** -0.5
        self.attention = Attention(num_heads, self.head_dim, num_kv_heads, self.self_attn_scale, max_position)

    def forward(
        self,
        x: torch.Tensor, # (num_tokens, hidden_size)
        positions: torch.Tensor, # (num_tokens, ) - position indices
    ): 
        # shape of x is (seq_len, hidden_size) phase1: considers only one sequence
        num_tokens, hidden_size = x.size()
        assert self.hidden_size == hidden_size

        # project x to qkv
        #TODO: may be transpose(1, 2) here to get shape (num_heads, num_tokens, head_dim)
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim) # (num_tokens, num_heads, head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)

        # k = (
        #     self.k_proj(x)
        #     .view(-1, 2, self.num_kv_heads, self.head_dim)
        #     .permute(1, 0, 2, 3)
        # )
        # v = (
        #     self.v_proj(x)
        #     .view(-1, 2, self.num_kv_heads, self.head_dim)
        #     .permute(1, 0, 2, 3)
        # )
        # k, v = kv.unbind(0)

        # apply qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # apply rope to q and k
        q, k = self.rope(positions, q, k)

        # apply Grouped Query Attention
        # attention with KV caching
        # Input: (num_tokens, num_heads, head_dim)
        # Output: (num_tokens, num_heads, head_dim)
        attn_output = self.attention(q, k, v) # attn_ouptut might not be contiguous

        # flatten heads back
        # (num_tokens, num_heads, head_dim) -> (num_tokens, num_heads * head_dim)
        attn_output = attn_output.reshape(-1, self.q_size) # using .reshape() cause attn_output might be non-contiguous

        output = self.o_proj(attn_output)

        return output   
    
class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor):
        """
        up = up_proj(x)

        gate_up = SiluMul(gate, up)
        down = down_proj(gate_up)
        SwiGLU variant
        """
        # gate = self.gate_proj(x)
        # up = self.up_proj(x) #(num_tokens, intermediate_size)
        # gate_up = torch.cat([gate, up], dim=-1)
        # x = self.act_fn(gate_up) # (num_tokens, intermediate_size)
        # x = self.down_proj(x) # (num_tokens, hidden_size)

        down_proj = self.down_proj(self.act_fn(torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)))
        return down_proj

class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()
        # attn layer
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # ffn layer
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # first layer, no residual yet
            # Just normalize hidden_sattes and save original as residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Subsequent layers: add residual inside norm
            # This uses RMSNorm's add_rms_forward method
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # apply attention
        hidden_states = self.self_attn(hidden_states, positions)
        # apply post-attention norm with residual
        # The residual from attention gets added here
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # apply ffn
        hidden_states = self.mlp(hidden_states)
        # return output and residual for next layer; residual adding after mlp happens in next layer
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()
        # print(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, 
        input_ids: torch.Tensor, # 
        positions: torch.Tensor,
    ):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 model with language modeling head.
    Final wrapper that adds logits generation.
    """
    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor, # (num_tokens, )
        positions: torch.Tensor, # (num_tokens, )
    ) -> torch.Tensor:
        
        hidden_states = self.model(input_ids, positions)
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        return logits
        