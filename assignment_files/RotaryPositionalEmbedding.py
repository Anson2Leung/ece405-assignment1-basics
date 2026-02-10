import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.d_k = d_k

        # 1/ (Θ (2k−2)/d)  k ∈ {1, . . . , d/2}
        # (2k-2)/d
        k = torch.arange(0, d_k, 2, device=device).float() / d_k
        angle = 1.0 / (theta ** k)

        # θ i,k
        i = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(i, angle)

        # 2d pre-computed buffer of sin and cos values created during init with self.register_buffer(persistent=False)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        # Create the 2d buffer for sin and cos
        self.register_buffer("sin_cached", torch.repeat_interleave(sin, 2, dim=-1), persistent=False)
        self.register_buffer("cos_cached", torch.repeat_interleave(cos, 2, dim=-1), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # use token positions to slice sin and cos
        sin = self.sin_cached[token_positions]
        cos = self.cos_cached[token_positions]
        x_left = x[..., 0::2]
        x_right = x[..., 1::2]
        
        # Interleave to get [-x2, x1, -x4, x3]
        # Flatten to maintain the pairwise interleaving (..., seq_len, d_k)
        x_rotated_base = torch.stack([-x_right, x_left], dim=-1).flatten(-2)

        # Apply the rotation formula: x*cos(theta) + x_rotated_base*sin(theta)
        return (x * cos) + (x_rotated_base * sin)