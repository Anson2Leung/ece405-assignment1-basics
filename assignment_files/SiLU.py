import torch
import torch.nn as nn
from .Linear import Linear

class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()

        # d_ff = 4 * d_model to approximately match SwiGLU parameter count
        if d_ff is None:
            d_ff = 4 * d_model

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(x) = x * sigmoid(x)
        gate_input = self.W1(x)
        activated  = gate_input * torch.sigmoid(gate_input)

        # down-projection
        return self.W2(activated)