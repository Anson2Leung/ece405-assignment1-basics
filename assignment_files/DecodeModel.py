import torch
from .tokenizer import Tokenizer
from .TransformerLM import TransformerLM
from .Softmax import softmax
from typing import Optional


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return softmax(logits / temperature, dim=-1)


def top_p_sampling(probs: torch.Tensor, p: float) -> torch.Tensor:
    # Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Cumulative sum to find cutoff
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative prob exceeds p
    # We shift right by 1 so we always include the token that pushes us over p
    sorted_indices_to_remove = (cumulative_probs - sorted_probs) >= p
    
    # Zero out tokens outside nucleus
    sorted_probs[sorted_indices_to_remove] = 0.0
    
    # Scatter back to original ordering and renormalize
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(0, sorted_indices, sorted_probs)
    filtered_probs = filtered_probs / filtered_probs.sum()
    
    return filtered_probs


@torch.no_grad()
def generate(
    model:          torch.nn.Module,
    prompt_tokens:  list[int],
    max_new_tokens: int,
    eos_token_id:   int,
    temperature:    float         = 1.0,
    top_p:          Optional[float] = None,
    device:         str           = "cpu",
) -> list[int]:
    
    model.eval()

    # Mutable copy of the prompt
    tokens = list(prompt_tokens)
    generated = []

    for _ in range(max_new_tokens):

        # Convert current context to tensor: (1, seq_len)
        input_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        # Forward pass logits shape: (1, seq_len, vocab_size)
        logits = model(input_tensor)

        # Take logits at the last position: (vocab_size,)
        next_token_logits = logits[0, -1, :]

        # Temperature scaling
        probs = softmax_with_temperature(next_token_logits, temperature)

        # Top-p / nucleus sampling
        if top_p is not None:
            probs = top_p_sampling(probs, top_p)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Stop
        if next_token == eos_token_id:
            break

        tokens.append(next_token)
        generated.append(next_token)

    return generated