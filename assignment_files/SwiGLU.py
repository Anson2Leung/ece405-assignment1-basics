import torch
import torch.nn as nn
import torch.nn.functional as F
from .Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        
        # d_ff: (8/3) * d_model, rounded to nearest multiple of 64
        # +32 to ensure rounding up to nearest multiple
        d_ff = int(8/3 * d_model)
        d_ff = 64 * ((d_ff + 32) // 64)
        
        # W1, W3 ∈ R ^dff×dmode
        # W2 ∈ R dmodel×dff
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W2(SiLU(W1x) ⊙ W3x)
        # SiLU(W1x)
        silu = F.silu(self.W1(x))

        # W3
        w3 = self.W3(x)

        # W2 (SiLU(W1x) ⊙ W3x)
        return self.W2(silu * w3)