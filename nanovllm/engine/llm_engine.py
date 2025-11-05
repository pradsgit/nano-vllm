# The main orchestrator class LLMEngine
import torch
from dataclasses import fields
from transformers import AutoTokenizer


from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.sampler import Sampler

class LLMEngine:
    def __init__(
        self,
        model: str,
        **kwargs,
    ):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # block_size = config.kvcache_block_size

        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config)
        self.sampler = Sampler()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.config = config

    def add_request(
        self, 
        prompt: str,
        sp: SamplingParams,
    ):
        """add the request to scheduler"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        seq = Sequence(prompt_tokens=prompt, sampling_params=sp)

        self.scheduler.add(seq)
        return seq
    

    def _sample(self, logits, sequences):
        temperatures = torch.tensor(
            [seq.sampling_params.temperature for seq in sequences],
            dtype=torch.float32,
            device=logits.device,
        )
        next_tokens_tensor = self.sampler(logits, temperatures)
        return next_tokens_tensor.tolist()
    
    def exit(self):
        import gc, torch

        # delete model_runner, model and k_cache and v_cache
        if hasattr(self.model_runner, 'model'):
            for layer in self.model_runner.model.model.layers:
                layer.self_attn.attention.k_cache = None
                layer.self_attn.attention.v_cache = None
                layer.self_attn.attention.num_cached_tokens = 0

            del self.model_runner.model
            del self.model_runner
        torch.cuda.empty_cache()

        gc.collect()
        print('\n\nGPU cleanup complete')

    def step(self):
        """
        executes one generation step
        """
        # schedule the sequences to run
        scheduled, num_prefill = self.scheduler.schedule()

        if not scheduled:
            return {}
        assert len(scheduled) == 1, "phase1 supports only single sequence"
        
        seq = scheduled[0]
        is_prefill = len(seq.output_tokens) == 0

        # run the model
        next_token = self.model_runner.run([seq], is_prefill)

        # if is_prefill:
        #     # Take last position's logits
        #     last_logit = logits[-1:, :]  # (1, vocab_size)
        # else:
        #     # decode already returns single logit
        #     last_logit = logits  # (1, vocab_size)

        # temperature = torch.tensor(
        #     [seq.sampling_params.temperature],
        #     dtype=torch.float32,
        #     device=logits.device,
        # )
        # next_token_tensor = self.sampler(last_logit, temperature)
        # next_token = next_token_tensor.item()

        # update sequence
        seq.add_token(next_token)

        # check if finsihed
        if self._is_finished(seq):
            seq.status = 'finished'
            self.scheduler.free_finished()

        return {seq.id: seq.output_tokens.copy()}

    
    def _is_finished(self, seq: Sequence):
        """
        check if sequence is finished generating
        """
        # 1. max tokens reached
        if len(seq.output_tokens) >= seq.sampling_params.max_tokens:
            return True
        
        # 2. eos token generated
        if hasattr(self.config, 'eos') and self.config.eos is not None:
            if seq.output_tokens and seq.output_tokens[-1] == self.config.eos:
                return True

    def generate(
        self, 
        prompts: list[str], # handling single request only for now
        sampling_params: SamplingParams,
    ):
        
        # if not isinstance(sampling_params, list):
        #     sampling_params = [sampling_params] * len(prompts)
        
        # for prompt, sp in zip(prompts, sampling_params):
        seq = self.add_request(prompts[0], sampling_params)

        # Run generation loop
        while seq.status != "finished":
            self.step()
        
        return seq.output_tokens
