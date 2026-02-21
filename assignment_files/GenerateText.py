import torch
import json
import argparse
from .TransformerLM import TransformerLM
from .tokenizer import Tokenizer
from .DecodeModel import generate
from .Checkpoint import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint",       type=str, required=True)
    parser.add_argument("--tokenizer_vocab",  type=str, required=True)
    parser.add_argument("--tokenizer_merges", type=str, required=True)
    parser.add_argument("--prompt",           type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens",   type=int, default=200)
    parser.add_argument("--temperature",      type=float, default=1.0)
    parser.add_argument("--top_p",            type=float, default=0.9)
    parser.add_argument("--device",           type=str, default="cuda:0")
    parser.add_argument("--vocab_size",       type=int, default=10000)
    parser.add_argument("--d_model",          type=int, default=512)
    parser.add_argument("--context_length",   type=int, default=256)
    parser.add_argument("--num_layers",       type=int, default=4)
    parser.add_argument("--num_heads",        type=int, default=16)
    parser.add_argument("--d_ff",             type=int, default=1344)
    parser.add_argument("--theta",            type=float, default=10000.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath  = args.tokenizer_vocab,
        merges_filepath = args.tokenizer_merges,
        special_tokens  = ["<|endoftext|>"]
    )

    # Create model model 
    model = TransformerLM(
        vocab_size     = args.vocab_size,
        context_length = args.context_length,
        d_model        = args.d_model,
        num_layers     = args.num_layers,
        num_heads      = args.num_heads,
        d_ff           = args.d_ff,
        theta          = args.theta,
        device         = str(device),
    ).to(device)

    # Load checkpoint 
    iteration = load_checkpoint(args.checkpoint, model, optimizer=None)
    print(f"Loaded checkpoint from iteration {iteration}")

    #  Encode prompt 
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    # Get EOS token ID 
    eos_token_id = tokenizer.vocab_reversed["<|endoftext|>".encode('utf-8')]

    # Generate
    generated_tokens = generate(
        model          = model,
        prompt_tokens  = prompt_tokens,
        max_new_tokens = args.max_new_tokens,
        eos_token_id   = eos_token_id,
        temperature    = args.temperature,
        top_p          = args.top_p,
        device         = str(device),
    )

    # Decode
    generated_text = tokenizer.decode(generated_tokens)
    print(f"\n--- Generated Text ---")
    print(args.prompt + generated_text)


if __name__ == "__main__":
    main()

# runs
"""
uv run python -m assignment_files.GenerateText \
    --checkpoint /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/lr_1e-3/ckpt_final.pt \
    --tokenizer_vocab  /home/ansonl32/ECE405/ece405-assignment1-basics/assignment_files/TinyStoriesVocab.json \
    --tokenizer_merges /home/ansonl32/ECE405/ece405-assignment1-basics/assignment_files/TinyStoriesMerges.json \
    --prompt          "Once upon a time" \
    --max_new_tokens  256 \
    --temperature     0.8 \
    --top_p           0.9 \
    --vocab_size      10000 \
    --device          cuda:0
"""
"""
uv run python -m assignment_files.GenerateText \
    --checkpoint /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/owt_1e3_10000_kill/ckpt_final.pt \
    --tokenizer_vocab  /home/ansonl32/ECE405/ece405-assignment1-basics/assignment_files/OWTVocab.json \
    --tokenizer_merges /home/ansonl32/ECE405/ece405-assignment1-basics/assignment_files/OWTMerges.json \
    --prompt          "My name is" \
    --max_new_tokens  256 \
    --temperature     0.8 \
    --top_p           0.9 \
    --vocab_size      32000 \
    --device          cuda:0
"""