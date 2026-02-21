#!/bin/bash
#SBATCH --job-name=ECE405_OWT_ks
#SBATCH --gres=gpu:1
#SBATCH --partition=kill-shared
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
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
    --max_steps       10000 \
    --warmup_steps    1000 \
    --lr              1e-3 \
    --beta1           0.9 \
    --beta2           0.999 \
    --weight_decay    0.1 \
    --grad_clip       1.0 \
    --min_lr_ratio    0.1 \
    --log_every       100 \
    --val_every       200 \
    --val_steps       20 \
    --checkpoint_every 5000 \
    --checkpoint_dir  /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/owt_1e3_10000_kill \
    --wandb_project   ece405 \
    --wandb_run_name  "owt_lr1e-3_10000_kill" \
    --ablation        none