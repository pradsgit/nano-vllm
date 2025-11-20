# The main orchestrator class LLMEngine
import torch
from dataclasses import fields
from transformers import AutoTokenizer
import time

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.model_runner import ModelRunner
# from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.sampler import Sampler
from nanovllm.utils.cache_context import GenerationMetrics

class LLMEngine:
    def __init__(
        self,
        model: str,
        **kwargs,
    ):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.model_runner = ModelRunner(config)
        self.scheduler = Scheduler(config)
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

            del self.model_runner.model
            del self.model_runner
        torch.cuda.empty_cache()

        gc.collect()
        print('\n\nGPU cleanup complete')

    def step(self):
        """
        schedule, run and extract output for one generation step
        """
        # schedule the sequences to run
        running_seqs, is_prefill = self.scheduler.schedule()

        if not running_seqs:
            return ValueError('No sequences to process for this step')
        
        # run the model
        next_tokens = self.model_runner.run(running_seqs, is_prefill)
        assert len(running_seqs) == len(next_tokens)
        self.scheduler.update_from_output(running_seqs, token_ids=next_tokens)

        # make output
        output = [
            (seq.id, seq.prompt_tokens.copy() + seq.output_tokens.copy())
            for seq in running_seqs
        ]
        
        return output, 1

    def is_finished(self):
        return self.scheduler.is_finished()

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
        prompts: list[str],
        sampling_params: SamplingParams,
    ):
        metrics = GenerationMetrics()

        # check if sampling_params is a list and its length is equal to prompts length
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        else:
            # is a list, its length must be equal to prompts length
            assert len(prompts) == len(sampling_params), "sampling_params list length does not match with prompts length"

        # we have list of prompts and their corresponding sampling_params
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}

        is_prefill = True

        while not self.is_finished():
            t0 = time.time()
            output, num_tokens = self.step()
            step_time = time.time() - t0

            if is_prefill:
                metrics.prefill_time = step_time
                is_prefill = False
            else:
                metrics.decode_times.append(step_time)

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids


        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        metrics.num_generated_tokens = sum([len(token_ids) for token_ids in outputs])
        metrics.print_summary()
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

