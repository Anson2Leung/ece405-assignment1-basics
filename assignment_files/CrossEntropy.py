import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(
    logits: Float[Tensor, "... vocab_size"], 
    targets: Int[Tensor, "..."]
) -> Float[Tensor, "..."]:
    """
    L = -log(softmax(logits)[target])
    L = -log( exp(x_y)/ sum( exp(x_i) ))
    L = -( log( exp(x_y) ) - log( sum( exp(x_i) )) )
    L = -x_y + log( sum( exp(x_i) )) 
    log( sum( exp(x_i) )) = M + log( sum( exp(x_i - M) ))
    note that M = max_logit
    x_y = true logit
    L = -logits[target] + log(sum(exp(logits - max_logit))) + max_logit
    """ 
    # Find max logit along the vocab dimension and subtract for stability
    # keeps dimensions for easy broadcasting
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits
    
    # log(sum(exp(x - max_logit))) + max_logit
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1)) + max_logits.squeeze(-1)
    
    # Get correct logit
    # gather expects index to have same number of dimensions as input
    true_class_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # -true_logit + log_sum_exp
    return -true_class_logits + log_sum_exp