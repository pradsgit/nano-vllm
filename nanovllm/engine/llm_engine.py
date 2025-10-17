# The main orchestrator class LLMEngine
import torch
from transformers import AutoTokenizer

from nanovllm.sampling_params import SamplingParams

class LLMEngine:
    def __init__(
        self,
        model: str,
    ):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    def generate(
        self, 
        prompt: str, # handling single request only for now
        sampling_params: SamplingParams,
    ):
        # what does generate do? usual case: prefill and decode phases of model forward

        # tokenize the prompt
        token_ids = self.tokenizer.encode(prompt)

        output = token_ids
        return output
