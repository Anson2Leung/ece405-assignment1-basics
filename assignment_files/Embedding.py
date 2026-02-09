import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Shape: (vocab_size, d_model)
        self.W = nn.Parameter(torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype
        ))

        # µ = 0, σ^2 = 1) truncated at [−3, 3]
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Translates token IDs into vectors using advanced indexing.
        Input token_ids: (batch_size, sequence_length) as LongTensor
        Output: (batch_size, sequence_length, embedding_dim)
        """
        # We index into the weight matrix. Each ID in token_ids acts as a row index.
        return self.W[token_ids]