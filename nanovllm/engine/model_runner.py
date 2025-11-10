import torch
import torch.nn as nn

from nanovllm.utils.sampler import Sampler
from nanovllm.model.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.cache_context import CacheContext, set_context

class ModelRunner:
    """
    load the model from huggingface, prepare inputs and run model forward pass
    """
    def __init__(
        self,
        config: Config,
    ):  
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size

        torch.set_default_device('cuda')
        torch.set_default_dtype(hf_config.torch_dtype)

        self.model = Qwen3ForCausalLM(hf_config)
        # load HF weights
        load_model(self.model, config.model)
        self.model = self.model.cuda()

        self.sampler = Sampler()

        print('move model to GPU')
        self.model = self.model.cuda()
        self.model.eval()
        self.init_kv_cache()


    def _prepare_inputs(
        self, 
        seqs: list[Sequence],
        is_prefill: bool,
    ):
        input_ids = []
        positions = []

        for seq in seqs:
            if is_prefill:
                tokens = seq.prompt_tokens
                pos = list(range(len(tokens)))
            else:
                # decode process only last generated token
                tokens = [seq.output_tokens[-1]]
                pos = [len(seq.prompt_tokens) + len(seq.output_tokens) - 1] # single position value

            input_ids.extend(tokens)
            positions.extend(pos)
        # pin_memory=True speeds up cpu-to-gpu movement
        return (
            torch.tensor(input_ids, dtype=torch.long, pin_memory=True, device='cuda'),
            torch.tensor(positions, dtype=torch.long, pin_memory=True, device='cuda')
        )

    def _prepare_inputs_for_prefill(
        self,
        seqs: list[Sequence],
    ):
        """
        for each sequence, build a "super sequence", global context and return input_ids and positions
        """
        raise NotImplementedError
    
    def _prepare_inputs_for_decode(
        self,
        seqs: list[Sequence],
    ):
        raise NotImplementedError
        
    
    def get_slot_mapping(
        self,
        block_ids: list[int],
        start_pos: int,
        end_pos: int, # exclusive
        block_size: int
    ) -> torch.Tensor:
        """
        Get slot mapping for logical token positions in [start_pos, end_pos).
        
        Works for:
        - Prefill: start=0, end=prompt_len
        - Decode:  start=L, end=L+1  (where L = current sequence length)
        
        Returns: tensor of shape (end_pos - start_pos,)
        """
        if start_pos >= end_pos:
            return torch.empty(0, dtype=torch.long, device='cuda')
        
        slots = []
        for logical_pos in range(start_pos, end_pos):
            block_idx = logical_pos // block_size # this gives relevant block id pos from block_ids list
            pos_in_block = logical_pos % block_size
            physical_block_id = block_ids[block_idx]
            slot = physical_block_id * block_size + pos_in_block
            slots.append(slot)
        
        return torch.tensor(slots, dtype=torch.long, device='cuda')


    def init_kv_cache(self):
        """
        allocate, reshape and bind KV cache tensors to attention layers
        """
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # calculate number of bytes for each block of kv_cache
        hf_config_dtype = hf_config.torch_dtype
        dtype_num_bytes = hf_config_dtype.itemsize
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype_num_bytes
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - peak - used + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # allocate a big ass empty tensor for kv caching
        self.kv_cache = torch.empty(
            2, 
            hf_config.num_hidden_layers, 
            config.num_kvcache_blocks, 
            self.block_size, 
            num_kv_heads, 
            head_dim,
            dtype=hf_config_dtype,
            device='cuda',
        )

        # inject k_cache and v_cache to Attention
        layer_idx = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = self.kv_cache[0][layer_idx].view(-1, num_kv_heads, head_dim)
                module.v_cache = self.kv_cache[1][layer_idx].view(-1, num_kv_heads, head_dim)
                layer_idx += 1
        
    @torch.inference_mode()
    def run_model(self, input_ids, positions):
        logits = self.model(input_ids, positions)
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool):
        # prepare inputs
        input_ids, positions = self._prepare_inputs(seqs, is_prefill)

        seq = seqs[0]
        # get slot mappings for currently processing token positions
        if is_prefill:
            slot_mapping = self.get_slot_mapping(
                seq.block_table,
                start_pos=0,
                end_pos=len(seq.prompt_tokens),
                block_size=self.block_size
            )
        else:
            # at decode time, we are processing the token at last position of sequnece
            current_pos = seq.num_cached_tokens
            slot_mapping = self.get_slot_mapping(
                seq.block_table,
                start_pos=current_pos,
                end_pos=current_pos + 1,
                block_size=self.block_size
            )

        # create context with sliced blocks
        cache_ctx = CacheContext(
            slot_mapping=slot_mapping,
            is_prefill=is_prefill,
            layer_idx=-1,
            block_table=seq.block_table,
            context_len=seq.num_cached_tokens,
            block_size=self.block_size
        )

        set_context(cache_ctx)

        logits = self.run_model(input_ids, positions)

        if is_prefill:
            last_logit = logits[-1, :].unsqueeze(0) # (1, vocab_size)
        else:
            last_logit = logits

        # run sampling to get the tokens
        temperature = torch.tensor(
            [seq.sampling_params.temperature],
            dtype=torch.float32,
            device=logits.device,
        )

        next_token_tensor = self.sampler(last_logit, temperature)
        next_token = next_token_tensor.item()

        return next_token
        
