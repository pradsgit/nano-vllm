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

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: Invalid logits detected")
            print(f"NaN count: {torch.isnan(logits).sum()}")
            print(f"Inf count: {torch.isinf(logits).sum()}")

        # Debug: check temperature values
        if (temperatures <= 0).any():
            print(f"ERROR: Invalid temperature: {temperatures}")
            raise ValueError("Temperature must be > 0")
        
        # temperature scaling
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = F.softmax(logits, dim=-1)

        # Debug: check probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"WARNING: Invalid probs after softmax")
            print(f"Probs NaN: {torch.isnan(probs).sum()}")
            print(f"Probs Inf: {torch.isinf(probs).sum()}")

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze_(-1)
        return next_tokens