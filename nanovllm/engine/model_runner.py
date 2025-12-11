import torch
import torch.nn as nn

from nanovllm.utils.sampler import Sampler
from nanovllm.model.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.cache_context import CacheContext, set_context, get_context

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
        self._init_kv_cache()


    def _prepare_inputs(
        self,
        seqs: list[Sequence],
    ):
        """
        for each sequence, build a "super sequence" (concatenated sequences), set global context and return input_ids and positions
        """
        input_ids = []
        positions = []
        slot_mapping = []

        cu_seqlens_q = [0]
        cu_seqlens_k = [0]

        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq in seqs:
            # if seq is prefill stage
            seqlen = len(seq)
            if seq.is_prefill:
                start = seq.num_computed_tokens
                end = start + seq.num_scheduled_tokens
                input_ids.extend(seq.prompt_tokens[start:end])
                positions.extend(list(range(start, end)))
                # get slot mapping for currently processing tokens
                slots = self.get_slot_mapping(
                    seq.block_table,
                    start_pos=start,
                    end_pos=end,
                    block_size=self.block_size
                )
            else:
                input_ids.append(seq.output_tokens[-1])
                positions.append(seq.num_tokens - 1)
                
                current_pos = seq.num_computed_tokens
                slots = self.get_slot_mapping(
                    seq.block_table,
                    start_pos=current_pos,
                    end_pos=current_pos+1,
                    block_size=self.block_size
                )

            slot_mapping.extend(slots)

            seqlen_q = seq.num_scheduled_tokens
            seqlen_k = seq.num_computed_tokens + seq.num_scheduled_tokens

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        block_tables = self._prepare_block_tables(seqs)

        ctx = CacheContext(
            layer_idx=-1,
            is_prefill=False,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_tables=block_tables,
        )
        set_context(ctx)
        return input_ids, positions

    def _prepare_block_tables(self, seqs: list[Sequence]):
        """pad to max length of block tables"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len-len(seq.block_table)) for seq in seqs] # right pad
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

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

    def _init_kv_cache(self):
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

        config.num_kvcache_blocks = 500
        # config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - peak - used + current) // block_bytes

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
                module.k_cache = self.kv_cache[0][layer_idx]
                module.v_cache = self.kv_cache[1][layer_idx]
                layer_idx += 1

    @torch.inference_mode()
    def run_model(self, input_ids, positions):
        logits = self.model(input_ids, positions)
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool):
        # prepare inputs for prefill and decode phase
        # here input_ids is a "super sequence". all the sequences are concatenated into a single seq
        input_ids, positions = self._prepare_inputs(seqs)

        ctx = get_context()

        logits = self.run_model(input_ids, positions) # logits shape: (num_tokens, vocab_size)
        
        # the logits contain seqs with partial prefill for which we do not want to sample tokens from,
        # only sample tokens from seqs that have processed full prefill or decode phase
        # how do you check for it? probably check seq.num_computed_tokens >= seq.prompt_tokens?
        
        # cu_seqlens_q = [0, 16, 29, 541, 1043]

        token_positions = ctx.cu_seqlens_q[1:] - 1 # get last token at each sequence ending position boundary 
        logits_sampling = logits[token_positions] # (num_seqs, vocab_size)

        # Identify which sequences completed prefill
        completed_prefill_mask = []
        for seq in seqs:
            total_processed = seq.num_computed_tokens + seq.num_scheduled_tokens
            is_complete = total_processed >= len(seq.prompt_tokens)
            completed_prefill_mask.append(is_complete)

        # Only sample for completed prefills
        temperature = torch.tensor(
            [seq.sampling_params.temperature for seq in seqs],
            dtype=torch.float32,
            device=logits.device,
        )

        all_tokens = self.sampler(logits_sampling, temperature).tolist()

        next_tokens = []
        for i, (seq, token, is_complete) in enumerate(zip(seqs, all_tokens, completed_prefill_mask)):
            if is_complete:
                next_tokens.append(token)
            else:
                next_tokens.append(None)  # Placeholder for partial prefill
        
        return next_tokens

        
