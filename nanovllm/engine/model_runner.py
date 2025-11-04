import torch
import torch.nn as nn

from nanovllm.utils.sampler import Sampler
from nanovllm.model.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence

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
        torch.set_default_device('cuda')
        torch.set_default_dtype(hf_config.torch_dtype)
        self.model = Qwen3ForCausalLM(hf_config)
        self.model = self.model.cuda()
        self.sampler = Sampler()
        # load HF weights
        load_model(self.model, config.model)

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

    @torch.inference_mode()
    def run_model(self, input_ids, positions):
        logits = self.model(input_ids, positions)
        return logits

    def run(self, seqs: list[Sequence], is_prefill: bool):
        # prepare inputs
        input_ids, positions = self._prepare_inputs(seqs)
        logits = self.run_model(input_ids, positions)

        if is_prefill:
            last_logit = logits[-1, :]
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
        
