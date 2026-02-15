import torch
import math
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr'] # alpha
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for theta in group['params']:
                if theta.grad is None:
                    continue

                grad = theta.grad.data
                state = self.state[theta]
                
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(theta.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(theta.data)
                
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']
                
                # Update First Moment (m)
                # m <- b_1 * m + (1 - beta1) * g
                # b_1 * m  +  alpha * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update Second Moment (v)
                # v <- b_2 * v + (1 - b_2) * g^2
                # b_2 * v  + (1-b_2)* grad*grad
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute Bias correction (alpha_t) 
                # α* √( 1−(β2)^t ) / ( 1−(β1)^t )
                correction = math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                alpha_t = lr * correction
                
                # Update Parameters
                #  θ − α_t * m/√v+ϵ
                # √v+ϵ = sqrt(v) + eps
                # theta + (-a_t * (m / denom))
                denom = v.sqrt().add_(eps)
                theta.data.addcdiv_(m, denom, value=-alpha_t)
                
                # Apply Weight Decay (Decoupled)
                # theta <- theta - alpha * lambda * theta
                # theta + (-lr * wd * theta)
                if wd > 0:
                    theta.data.add_(theta.data, alpha=-lr * wd)
                    
        return loss