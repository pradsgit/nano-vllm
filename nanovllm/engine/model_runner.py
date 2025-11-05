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
        load_model(self.model, config.model)
        self.model = self.model.cuda()
        self.sampler = Sampler()
        # load HF weights

        self.allocate_kv_cache()

        print('move model to GPU')
        self.model = self.model.cuda()
        self.model.eval()

    def _prepare_inputs(
        self, 
        seqs: list[Sequence], 
    ):
        input_ids = []
        positions = []

        for seq in seqs:
            if len(seq.output_tokens) == 0:
                # this is prefill phase
                tokens = seq.prompt_tokens
                pos = list(range(len(tokens)))
            else:
                # decode process only last generated token
                tokens = [seq.output_tokens[-1]]
                pos = [len(seq.prompt_tokens) + len(seq.output_tokens) - 1] # single position value

            input_ids.extend(tokens)
            positions.extend(pos)

        # convert to tensors
        # pin_memory=True speeds up cpu-to-gpu movement
        return (
            torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(), 
            torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda()
        )
    
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()
        # used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # calculate number of bytes for each block of kv_cache
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - peak) // block_bytes
        assert config.num_kvcache_blocks > 0
        # allocate a big ass empty tensor
        self.kv_cache = torch.empty(
            2, 
            hf_config.num_hidden_layers, 
            config.num_kvcache_blocks, 
            self.block_size, 
            num_kv_heads, 
            head_dim,
            dtype=hf_config.torch_dtype,
            device='cuda',
        )
        
    @torch.inference_mode()
    def run_model(self, input_ids, positions):
        logits = self.model(input_ids, positions)
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool):
        # prepare inputs
        input_ids, positions = self._prepare_inputs(seqs)

        seq = seqs[0]
        block_ids = seq.block_table # eg: [3, 20, 18]

        # get relevant kv_blocks from self.kv_cache and save it to cache_context
        kv_blocks = [
            self.kv_cache[:, :, id, :, :, :] # slice for each block
            for id in block_ids
        ]

        slot_idx = 0 if is_prefill else seq.num_tokens

        # create context with sliced blocks
        cache_ctx = CacheContext(
            kv_blocks=kv_blocks,
            slot_idx=slot_idx,
            is_prefill=is_prefill,
            layer_idx=-1,
        )

        set_context(cache_ctx)

        logits = self.run_model(input_ids, positions)

        if is_prefill:
            last_logit = logits[-1, :].unsqueeze(0) # (1, vocab_size)
        else:
            last_logit = logits

        # run sampling to get the tokens
        temperature = torch.tensor(
            [seqs[0].sampling_params.temperature],
            dtype=torch.float32,
            device=logits.device,
        )

        next_token_tensor = self.sampler(last_logit, temperature)
        next_token = next_token_tensor.item()

        return next_token
        
