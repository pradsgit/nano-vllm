import torch
import torch.nn.functional as F
from nanovllm.sampling_params import SamplingParams

class Sampler:
    """sample next tokens from logits"""
    def __call__(
        self, 
        logits: torch.Tensor, # (num_tokens, vocab_size)
        temperatures: torch.Tensor,
        # sampling_params: SamplingParams
    ) -> list[int]:
        
        # temperature scaling
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # softmax
        probs = F.softmax(logits, dim=-1)
        # sampling from multinomial
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_tokens