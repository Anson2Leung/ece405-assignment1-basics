import math

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int) -> float:
    
    # Warm-up Phase
    if t < t_w:
        return (t / t_w) * alpha_max
        
    # Cosine Annealing Phase
    elif t <= t_c:
        decay_ratio = (t - t_w) / (t_c - t_w)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        # Scale it between the max and min learning rates
        return alpha_min + coefficient * (alpha_max - alpha_min)
        
    # Post-annealing Phase
    else:
        return alpha_min