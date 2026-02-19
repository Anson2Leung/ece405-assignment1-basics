import torch
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    eps = 1e-6
    
    # Collect gradients that exist
    grads = [p.grad for p in parameters if p.grad is not None]
    
    # Compute global L2 norm 
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    
    # Scale down if norm exceeds max_norm
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)