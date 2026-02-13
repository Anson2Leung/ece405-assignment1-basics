import torch
import torch.nn as nn
from .TransformerBlock import TransformerBlock
from .RMSNorm import RMSNorm
from .Linear import Linear
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding

class TransformerLM(nn.Module):
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
        
        # Token Embeddings
        # Converts input token IDs to d_model vectors
        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device)
        
        # Rotary Positional Embeddings (RoPE)
        # Shared across all layers. Calculated once based on context_length.
        self.rope = RotaryPositionalEmbedding(
            d_k=d_model // num_heads, 
            theta=theta, 
            max_seq_len=context_length, 
            device=device
        )
        
        # Create TransformerBlock layers
        # ModuleList to register these sub-modules properly
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device)
            for _ in range(num_layers)
        ])
        
        # Apply Norm and Linear
        self.final_norm = RMSNorm(d_model, device=device)
        self.output_layer = Linear(d_model, vocab_size, device=device)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # Generate token positions based on actual sequence length
        seq_len = in_indices.shape[1]
        token_positions = torch.arange(seq_len, device=in_indices.device)
        
        # Embed tokens
        x = self.token_embeddings(in_indices)
        
        # Transformer Blocks
        for layer in self.layers:
            # We pass the shared rope and positions to every block
            x = layer(x, rope=self.rope, token_positions=token_positions)
            
        # Final Norm
        x = self.final_norm(x)
        
        # Project to Vocabulary (Logits)
        logits = self.output_layer(x)
        
        return logits