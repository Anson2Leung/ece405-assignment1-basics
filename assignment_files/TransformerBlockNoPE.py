import torch
import torch.nn as nn
from .MultiHeadSelfAttentionNoPE import MultiHeadSelfAttentionNoPE
from .SwiGLU import SwiGLU
from .RMSNorm import RMSNorm

class TransformerBlockNoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None):
        super().__init__()
        self.norm_1 = RMSNorm(d_model, device=device)
        self.attn   = MultiHeadSelfAttentionNoPE(d_model, num_heads, device=device)
        self.norm_2 = RMSNorm(d_model, device=device)
        self.ffn    = SwiGLU(d_model, d_ff=d_ff, device=device)

    def forward(self, x: torch.Tensor, rope: nn.Module = None, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # no RoPE
        y = x + self.attn(self.norm_1(x), mask=mask)
        z = y + self.ffn(self.norm_2(y))
        return z