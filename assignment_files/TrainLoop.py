import os
import time
import argparse
import numpy as np
import torch
import wandb
import logging
import json

from typing import Optional
from .TransformerLM import TransformerLM
from .AdamW import AdamW
from .DataLoader import get_batch
from .Checkpoint import save_checkpoint, load_checkpoint
from .GradientClipping import gradient_clipping
from .LearningRate import lr_cosine_schedule
from .CrossEntropy import cross_entropy

# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data
    parser.add_argument("--train_data",      type=str, required=True,  help="path to training data file")
    parser.add_argument("--val_data",        type=str, required=True)

    # Model hyperparameters
    parser.add_argument("--vocab_size",     type=int, default=32000)
    parser.add_argument("--d_model",        type=int, default=512)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--num_layers",     type=int, default=6)
    parser.add_argument("--num_heads",      type=int, default=8)
    parser.add_argument("--d_ff",           type=int, default=2048)
    parser.add_argument("--theta",          type=float, default=10000.0)

    # Optimizer hyperparameters
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--beta1",        type=float, default=0.9)
    parser.add_argument("--beta2",        type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip",    type=float, default=1.0)

    # LR schedule
    parser.add_argument("--warmup_steps",  type=int,   default=2000)
    parser.add_argument("--max_steps",     type=int,   default=10000)
    parser.add_argument("--min_lr_ratio",  type=float, default=0.1)

    # Training settings
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--dtype",      type=str, default="float32")

    # Checkpointing
    parser.add_argument("--checkpoint_dir",   type=str, default="checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--resume_from",      type=str, default=None)

    # Logging
    parser.add_argument("--log_every",      type=int, default=100)
    parser.add_argument("--val_every",      type=int, default=200)
    parser.add_argument("--val_steps",      type=int, default=20)
    parser.add_argument("--wandb_project",  type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    #Ablations
    parser.add_argument("--ablation", type=str, default="none", 
                        choices=["none", "no_norm", "post_norm", "nope", "silu"], 
                        help="Set Ablation"
)

    return parser.parse_args()


@torch.no_grad()
def estimate_val_loss(
    model:          torch.nn.Module,
    val_data:       np.ndarray,
    batch_size:     int,
    context_length: int,
    device:         str,
    val_steps:      int,
) -> float:
    """Estimate validation loss over several batches."""
    model.eval()
    losses = []
    for _ in range(val_steps):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y).mean()
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(
    model:           torch.nn.Module,
    optimizer:       torch.optim.Optimizer,
    train_data:      np.ndarray,
    val_data:        np.ndarray,
    args:            argparse.Namespace,
    start_iteration: int = 0,
) -> None:

    model.train()
    min_lr = args.lr * args.min_lr_ratio
    logger.info("Starting training from step %d to %d", start_iteration, args.max_steps)

    # --- Wallclock tracking ---
    train_start_time = time.time()
    step_start_time  = time.time()

    for step in range(start_iteration, args.max_steps):

        if torch.isnan(model.token_embeddings.weight).any():
            logger.warning("NaN in embeddings at step %d, stopping run", step)
            break
        
        # --- LR schedule ---
        lr = lr_cosine_schedule(
            t                  = step,
            alpha_max          = args.lr,
            alpha_min          = min_lr,
            t_w                = args.warmup_steps,
            t_c                = args.max_steps,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        
        x, y   = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss   = cross_entropy(logits, y).mean()

        # NaN check
        if torch.isnan(loss):
            logger.warning("NaN loss at step %d, stopping run", step)
            break
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        
        # Check for NaN gradients
        nan_gradient_found = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.warning("NaN gradient in %s at step %d", name, step)
                nan_gradient_found = True
                break
        if nan_gradient_found:
            break
        
        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        # Training logging 
        if step % args.log_every == 0:
            ppl          = float(np.exp(loss.item()))
            elapsed      = time.time() - train_start_time        # total wallclock
            step_time    = (time.time() - step_start_time) / max(step - start_iteration, 1)  # avg seconds/step
            tokens_seen  = step * args.batch_size * args.context_length

            logger.info(
                "step %6d | loss %.4f | ppl %.2f | lr %.2e | "
                "elapsed %.1fs | ms/step %.1f | tokens %s",
                step, loss.item(), ppl, lr,
                elapsed, step_time * 1000, f"{tokens_seen:,}"
            )

            if args.wandb_project:
                wandb.log({
                    "train/loss":      loss.item(),
                    "train/ppl":       ppl,
                    "lr":              lr,
                    "wallclock_time":  elapsed,
                    "ms_per_step":     step_time * 1000,
                    "tokens_seen":     tokens_seen,
                }, step=step)

        # Validation
        if step % args.val_every == 0:
            val_loss = estimate_val_loss(
                model, val_data, args.batch_size,
                args.context_length, args.device, args.val_steps,
            )
            val_ppl  = float(np.exp(val_loss))
            elapsed  = time.time() - train_start_time

            logger.info(
                "  >> val loss %.4f | val ppl %.2f | wallclock %.1fs",
                val_loss, val_ppl, elapsed
            )

            if args.wandb_project:
                wandb.log({
                    "val/loss":       val_loss,
                    "val/ppl":        val_ppl,
                    "wallclock_time": elapsed,
                }, step=step)

        # Checkpointing
        if step % args.checkpoint_every == 0 and step > start_iteration:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            logger.info("  >> Checkpoint saved to %s", ckpt_path)

    # Final checkpoint 
    total_time = time.time() - train_start_time
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_steps, final_path)
    logger.info(
        "Training complete in %.1fs (%.2f hrs). Final checkpoint saved to %s",
        total_time, total_time / 3600, final_path
    )


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype  = torch.float16 if args.dtype == "float16" else torch.float32
    logger.info("Using device: %s  |  dtype: %s", device, dtype)

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        logger.info("W&B run initialised: project=%s", args.wandb_project)

    # Data
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data   = np.memmap(args.val_data,   dtype=np.uint16, mode='r')
    logger.info("Train tokens: %s  |  Val tokens: %s",
                f"{len(train_data):,}", f"{len(val_data):,}")

    # Model / ablation selection
    if args.ablation == "none":
        model = TransformerLM(
            vocab_size     = args.vocab_size,
            context_length = args.context_length,
            d_model        = args.d_model,
            num_layers     = args.num_layers,
            num_heads      = args.num_heads,
            d_ff           = args.d_ff,
            theta          = args.theta,
            device         = args.device,
        ).to(dtype=dtype)
    elif args.ablation == "no_norm":
        from .TransformerLMNoNorm import TransformerLMNoNorm
        model = TransformerLMNoNorm(
            vocab_size     = args.vocab_size,
            context_length = args.context_length,
            d_model        = args.d_model,
            num_layers     = args.num_layers,
            num_heads      = args.num_heads,
            d_ff           = args.d_ff,
            theta          = args.theta,
            device         = args.device,
        ).to(dtype=dtype)
        logger.info("Using TransformerLM WITHOUT RMSNorm (ablation)")
    elif args.ablation == "post_norm":
        from .TransformerLMPostNorm import TransformerLMPostNorm
        model = TransformerLMPostNorm(
            vocab_size     = args.vocab_size,
            context_length = args.context_length,
            d_model        = args.d_model,
            num_layers     = args.num_layers,
            num_heads      = args.num_heads,
            d_ff           = args.d_ff,
            theta          = args.theta,
            device         = args.device,
        ).to(dtype=dtype)
        logger.info("Using TransformerLM with POST-norm")
    elif args.ablation == "nope":
        from .TransformerLMNoPE import TransformerLMNoPE
        model = TransformerLMNoPE(
            vocab_size     = args.vocab_size,
            context_length = args.context_length,
            d_model        = args.d_model,
            num_layers     = args.num_layers,
            num_heads      = args.num_heads,
            d_ff           = args.d_ff,
            theta          = args.theta,
            device         = args.device,
        ).to(dtype=dtype)
        logger.info("Using TransformerLM no positional embeddings (NoPE)")
    elif args.ablation == "silu":
        from .TransformerLMSiLU import TransformerLMSiLU
        model = TransformerLMSiLU(
            vocab_size     = args.vocab_size,
            context_length = args.context_length,
            d_model        = args.d_model,
            num_layers     = args.num_layers,
            num_heads      = args.num_heads,
            d_ff           = args.d_ff,
            theta          = args.theta,
            device         = args.device,
        ).to(dtype=dtype)
        logger.info("Using TransformerLM with SiLU FFN (d_ff=4*d_model)")

    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{num_params:,}")

    # AdamW
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        betas        = (args.beta1, args.beta2),
        weight_decay = args.weight_decay,
    )

    # Resume from checkpoint if provided
    start_iteration = 0
    if args.resume_from:
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        logger.info("Resumed from checkpoint at iteration %d", start_iteration)

    # Train
    train(
        model           = model,
        optimizer       = optimizer,
        train_data      = train_data,
        val_data        = val_data,
        args            = args,
        start_iteration = start_iteration,
    )

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
