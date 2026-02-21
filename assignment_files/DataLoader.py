import numpy as np
import torch

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample random starting indices
    # (1 ≤ i < n − m )
    indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    
    # Gather input and target sequences
    inputs  = np.stack([x[i : i + context_length]     for i in indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in indices])
    
    # Convert to tensors containing token IDs and move to device
    inputs  = torch.tensor(inputs,  dtype=torch.long).to(device)
    targets = torch.tensor(targets, dtype=torch.long).to(device)
    
    return inputs, targets