import torch
import torch.nn as nn
from .Linear import Linear
from .Softmax import scaled_dot_product_attention

class MultiHeadSelfAttentionNoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_qkv     = Linear(d_model, 3 * d_model, device=device)
        self.W_o       = Linear(d_model, d_model, device=device)

    def forward(self, x: torch.Tensor, rope: nn.Module = None, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.W_qkv(x)
        Q, K, V = torch.split(qkv, self.d_model, dim=-1)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # No RoPE applied to Q and K

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(attn_out)