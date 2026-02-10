import torch
import torch.nn as nn
import math
from .Linear import Linear
from .Softmax import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # combining the key, query, and value projections
        # Using 3 * d_model to hold Q, K, and V
        self.W_qkv = Linear(d_model, 3 * d_model, device=device)

        # WOMultiHead(WQx, WKx, WVx)
        self.W_o = Linear(d_model, d_model, device=device)

    def forward(self, x: torch.Tensor, rope: nn.Module, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # batch size, sequence length,  embedding dimensions
        batch_size, seq_len, _ = x.shape
        
        # Combining the key, query, and value projections
        qkv = self.W_qkv(x) 

        # the same RoPE rotation should be applied to the query and key vectors for each head
        Q, K, V = torch.split(qkv, self.d_model, dim=-1)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE if provided
        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        # use causal attention masking, triu is the wrong side of the mask
        if mask is None:
            #mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            
        
        # Softmax Scaled Dot-Product Attention
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # MultiHead(Q, K, V ) = Concat(head1, . . . , headh)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_o(attn_out)