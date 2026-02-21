import torch
import torch.nn as nn
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .SwiGLU import SwiGLU
from .RMSNorm import RMSNorm

class TransformerBlockPostNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None
    ):
        super().__init__()
        self.attn   = MultiHeadSelfAttention(d_model, num_heads, device=device)
        self.norm_1 = RMSNorm(d_model, device=device)
        self.ffn    = SwiGLU(d_model, d_ff=d_ff, device=device)
        self.norm_2 = RMSNorm(d_model, device=device)

    def forward(self, x: torch.Tensor, rope: nn.Module, token_positions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Post-norm
        # z = RMSNorm(x + MHSA(x))
        # y = RMSNorm(z + FFN(z))
        z = self.norm_1(x + self.attn(x, rope=rope, mask=mask, token_positions=token_positions))
        y = self.norm_2(z + self.ffn(z))
        return y