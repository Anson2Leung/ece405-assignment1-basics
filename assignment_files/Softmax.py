import torch
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # we will subtract the largest entry of oi from all elements of oi, making the new largest entry 0.
    max_val = torch.max(x, dim=dim, keepdim=True).values
    
    # Softmax softmax(v)i = exp(vi)/( ∑ (n, j=1) exp(vj)
    exps = torch.exp(x - max_val)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    return exps / sum_exps

def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    mask: torch.Tensor = None
) -> torch.Tensor:
    # Attention(Q, K, V ) = softmax ( Q⊤K/ sqrt(dk) )V
    d_k = query.size(-1)

    # Q ⊤ranspose K
    # (..., n, d_k) @ (..., d_k, m) -> (..., n, m)
    scores = torch.einsum('...nd, ...md -> ...nm', query, key)

    # 1/sqrt(dk)
    scores = scores / math.sqrt(d_k)

    # Apply mask, mask=False means "do not attend"
    # add -infinity of mask matrix that is False
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    attention_weights = softmax(scores, dim=-1)

    # (..., n, m) @ (..., m, d_v) -> (..., n, d_v)
    return torch.einsum('...nm, ...md -> ...nd', attention_weights, value)