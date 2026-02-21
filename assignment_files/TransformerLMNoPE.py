import torch
import torch.nn as nn
from .TransformerBlockNoPE import TransformerBlockNoPE
from .RMSNorm import RMSNorm
from .Linear import Linear

class TransformerLMNoPE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = None,
        device=None
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device)

        self.layers = nn.ModuleList([
            TransformerBlockNoPE(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device)
            for _ in range(num_layers)
        ])

        self.final_norm   = RMSNorm(d_model, device=device)
        self.output_layer = Linear(d_model, vocab_size, device=device)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)   # no rope or token_positions passed

        x = self.final_norm(x)
        return self.output_layer(x)