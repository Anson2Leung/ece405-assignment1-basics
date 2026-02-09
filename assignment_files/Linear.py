import torch
import torch.nn as nn
import numpy as np
from einops import einsum
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/modules/linear.py#

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # construct and store your parameter as W putting it in an nn.Parameter
        self.W = nn.Parameter(torch.empty(
            (out_features, in_features), 
            device=device, 
            dtype=dtype
        ))

        # µ = 0, σ^2 =2/(din+dout ) truncated at [−3σ, 3σ]
        sigma = np.sqrt((2/in_features+out_features))

        # use the settings from above along with torch.nn.init.trunc_normal_ to initialize the weights.
        nn.init.trunc_normal_(
            self.W, 
            mean=0.0, 
            std=sigma,
            a=-3.0*sigma, 
            b=3.0*sigma 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = Wx.
        #return einsum(x, self.W, "batch sequence d_in, d_out d_in -> batch sequence d_out")
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")