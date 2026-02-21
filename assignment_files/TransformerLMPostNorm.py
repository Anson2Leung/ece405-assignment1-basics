import torch
import torch.nn as nn
from .TransformerBlockPostNorm import TransformerBlockPostNorm
from .RMSNorm import RMSNorm
from .Linear import Linear
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding

class TransformerLMPostNorm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device)

        self.rope = RotaryPositionalEmbedding(
            d_k=d_model // num_heads,
            theta=theta,
            max_seq_len=context_length,
            device=device
        )

        self.layers = nn.ModuleList([
            TransformerBlockPostNorm(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device)
            for _ in range(num_layers)
        ])

        # Post-norm keeps final norm
        self.final_norm = RMSNorm(d_model, device=device)
        self.output_layer = Linear(d_model, vocab_size, device=device)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        seq_len = in_indices.shape[1]
        token_positions = torch.arange(seq_len, device=in_indices.device)

        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)

        x = self.final_norm(x)
        return self.output_layer(x)