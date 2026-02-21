import torch
import torch.nn as nn
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .SiLU import SiLUFFN
from .RMSNorm import RMSNorm

class TransformerBlockSiLU(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, device=None):
        super().__init__()
        self.norm_1 = RMSNorm(d_model, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device)
        self.norm_2 = RMSNorm(d_model, device=device)
        # d_ff = 4 * d_model
        self.ffn = SiLUFFN(d_model, d_ff=4*d_model, device=device)

    def forward(self, x: torch.Tensor, rope: nn.Module, token_positions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        y = x + self.attn(self.norm_1(x), rope=rope, mask=mask, token_positions=token_positions)
        z = y + self.ffn(self.norm_2(y))
        return z