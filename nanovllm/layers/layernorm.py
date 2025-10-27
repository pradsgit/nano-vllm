# hidden_size, gain param of hidden_size size
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    computes x -> w * x / sqrt(E[x^2] + eps)
    """
    def __init__(
        self, 
        hidden_size: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))

    @torch.compile
    def rms_forward(self, x:torch.Tensor) -> torch.Tensor:
        """RMSNorm should be computed in float32"""
        x_dtype = x.dtype
        # upcast to float32 to prevent overflow when squared
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x.mul_(rms)
        x = x.to(x_dtype)
        x.mul_(self.weight.to(x_dtype))
        return x
    
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # what exactly happens here?
        # residual is added inside the next normalization layer instead of immediately.
        # The additions don't happen "after" sublayers - they happen "before the next norm" instead.
        x_dtype = x.dtype
        x = x.float().add_(residual.float())
        # now x has residual info
        residual = x.to(x_dtype )
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x.mul_(rms)
        x = x.to(x_dtype)
        x.mul_(self.weight.to(x_dtype))
        return x, residual

    def forward(
        self, 
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)