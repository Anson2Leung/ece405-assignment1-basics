import torch
import torch.nn as nn
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .SwiGLU import SwiGLU

class TransformerBlockNoNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        device=None
    ):
        super().__init__()
        # No RMSNorm — just MHSA and FFN
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device)
        self.ffn  = SwiGLU(d_model, d_ff=d_ff, device=device)

    def forward(self, x: torch.Tensor, rope: nn.Module, token_positions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # No norm before attention or FFN
        y = x + self.attn(x, rope=rope, mask=mask, token_positions=token_positions)
        z = y + self.ffn(y)
        return z