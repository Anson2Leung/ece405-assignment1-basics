#!/bin/bash
#SBATCH --job-name=ECE405_7.2base
#SBATCH --gres=gpu:1
#SBATCH --partition=kill-shared 
#SBATCH --time=12:00:00 
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --error=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/train_error%j.log
#SBATCH --output=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/train%j.log

# remember to insert and remove
export WANDB_API_KEY=""

# Filepath locations
TOK_TRAIN="/home/ansonl32/ECE405/data/tinystories_train.npy"
TOK_VALID="/home/ansonl32/ECE405/data/tinystories_valid.npy"

# LR's to test[3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3]
LR=${2:-3e-5}
# 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128
BATCH_SIZE=${1:-80}

cd /home/ansonl32/ECE405/ece405-assignment1-basics

uv run python -m assignment_files.TrainLoop \
    --train_data      "$TOK_TRAIN" \
    --val_data        "$TOK_VALID" \
    --vocab_size      10000 \
    --d_model         512 \
    --context_length  256 \
    --num_layers      4 \
    --num_heads       16 \
    --d_ff            1344 \
    --theta           10000 \
    --batch_size      "$BATCH_SIZE" \
    --max_steps       16000 \
    --warmup_steps    1600 \
    --lr              "$LR" \
    --beta1           0.9 \
    --beta2           0.999 \
    --weight_decay    0.1 \
    --grad_clip       1.0 \
    --min_lr_ratio    0.1 \
    --log_every       100 \
    --val_every       200 \
    --val_steps       20 \
    --checkpoint_every 2000 \
    --checkpoint_dir  /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/tinystories_base \
    --wandb_project   ece405 \
    --wandb_run_name  "bs${BATCH_SIZE}_lr${LR}"