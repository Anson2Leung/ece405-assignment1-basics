import torch
import torch.nn as nn
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .SwiGLU import SwiGLU
from .RMSNorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        # Norm + MHSA w/ Rope
        self.norm_1 = RMSNorm(d_model, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device)
        
        # Norm + Position wise Feed Forward network
        self.norm_2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device)

    def forward(self, x: torch.Tensor, rope: nn.Module, mask: torch.Tensor = None) -> torch.Tensor:
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        # y = x + FFN(RMSNorm(x))
        y = x + self.attn(self.norm_1(x), rope=rope, mask=mask)
        z = y + self.ffn(self.norm_2(y))
        
        return z