import torch
import torch.nn as nn
import torch.nn.functional as F
from .Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        
        # 1. Calculate d_ff if not provided, ensuring multiple of 64
        if d_ff is None:
            d_ff = int(8/3 * d_model)
            d_ff = 64 * ((d_ff + 32) // 64)
        
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(x) = x * sigmoid(x)
        gate_input = self.W1(x)
        swish_gate = gate_input * torch.sigmoid(gate_input)
        
        # element-wise product with the up-projection
        up_proj = self.W3(x)
        intermediate = swish_gate * up_proj
        
        # down-projection
        return self.W2(intermediate)