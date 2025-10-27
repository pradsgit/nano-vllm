# implements SiluAndMul activation
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # d = x.size(-1) // 2
        # x, y = x[..., :d], x[..., d:]
        x, y = x.chunk(2, dim=-1) # break in to two chunks at last dimension

        return F.silu(x) * y
