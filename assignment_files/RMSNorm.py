import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Your code here performing RMSNorm
        #  RMS(a) = sqrt( (1/d_model) * sum(i->d_model: ai^2) + epsilon )
        # (1/d_model) * sum(i->d_model: ai^2) -> mean (ai^2)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # RMSNorm
        # (a / rms(a)) * g
        result = (x / rms) * self.g

        # downcast to original type
        return result.to(in_dtype)