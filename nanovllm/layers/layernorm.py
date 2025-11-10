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
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=None, device=device)
        )

    @torch.compile
    def rms_forward(self, x:torch.Tensor) -> torch.Tensor:
        """RMSNorm should be computed in float32"""
        x_dtype = x.dtype
        # upcast to float32 to prevent overflow when squared
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # back to its original dtype
        x.mul_(rms)
        x = x.to(x_dtype).mul_(self.weight)
        return x
    
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
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
        

class RMSNorm(nn.Module):
    """
    Computes x -> w * x / sqrt(E[x^2] + eps)
    """
    def __init__(
        self,
        hidden_size: int, 
        eps: float = 1e-6,
        dtype: torch.dtype | None = None, 
        device: torch.device | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
    
    @torch.compile
    def forward(
        self,
        x: torch.Tensor, # (num_tokens, hidden_size)
        residual: torch.Tensor | None = None# (num_tokens, hidden_size)
    ):
        # RMSNorm always upcasts inputs to float32 for stability
        x_dtype = x.dtype

        # add residual if provided
        if residual is not None:
            x = x.float().add_(residual) # x.float() will create a new tensor if and only if x dtype is not float32
            new_residual = x.to(x_dtype)

        # compute rms in float32
        x = x if residual is not None else x.float()
        # calculate rms scaling factor 
        rms_factor = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # (num_tokens, 1)
        x.mul_(rms_factor) # in-place op # (num_tokens, hidden_size)
        # in qwen3, the multiplication with the weight is done on the original dtype tensor
        x = x.to(x_dtype).mul_(self.weight)

        return (x, new_residual) if residual is not None else x