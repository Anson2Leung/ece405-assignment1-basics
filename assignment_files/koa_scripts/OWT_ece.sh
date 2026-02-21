#!/bin/bash
#SBATCH --job-name=ECE405_OWT
#SBATCH --gres=gpu:1
#SBATCH --account=ece405
#SBATCH --partition=ece405
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --error=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/owt_error%j.log
#SBATCH --output=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/owt%j.log

export WANDB_API_KEY=""

# OWT tokenized data
TOK_TRAIN="/home/ansonl32/ECE405/data/owt_train.npy"
TOK_VALID="/home/ansonl32/ECE405/data/owt_valid.npy"

cd /home/ansonl32/ECE405/ece405-assignment1-basics

uv run python -m assignment_files.TrainLoop \
    --train_data      "$TOK_TRAIN" \
    --val_data        "$TOK_VALID" \
    --vocab_size      32000 \
    --d_model         512 \
    --context_length  256 \
    --num_layers      4 \
    --num_heads       16 \
    --d_ff            1344 \
    --theta           10000 \
    --batch_size      128 \
    --max_steps       5000 \
    --warmup_steps    500 \
    --lr              64 \
    --beta1           0.9 \
    --beta2           0.999 \
    --weight_decay    0.1 \
    --grad_clip       1.0 \
    --min_lr_ratio    0.1 \
    --log_every       100 \
    --val_every       200 \
    --val_steps       20 \
    --checkpoint_every 2500 \
    --checkpoint_dir  /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/owt_base \
    --wandb_project   ece405 \
    --wandb_run_name  "owt_lr64" \
    --ablation        none