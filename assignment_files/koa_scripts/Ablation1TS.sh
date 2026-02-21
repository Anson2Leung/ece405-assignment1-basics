#!/bin/bash
#SBATCH --job-name=Ablation_nonorm
#SBATCH --gres=gpu:1
#SBATCH --account=ece405
#SBATCH --partition=ece405
#SBATCH --time=6:00:00 
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --error=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/nonorm2_error%j.log
#SBATCH --output=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/nonorm2_%j.log

# remember to insert and remove
export WANDB_API_KEY=""

export CUDA_LAUNCH_BLOCKING=1

# Filepath locations
TOK_TRAIN="/home/ansonl32/ECE405/data/tinystories_train.npy"
TOK_VALID="/home/ansonl32/ECE405/data/tinystories_valid.npy"

# Switch out LR values 
cd /home/ansonl32/ECE405/ece405-assignment1-basics

for LR in 6e-4 1e-3; do
    echo "========================================="
    echo "Starting run with LR=$LR"
    echo "========================================="
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
        --batch_size      80 \
        --max_steps       5000 \
        --warmup_steps    500 \
        --lr              "$LR" \
        --beta1           0.9 \
        --beta2           0.999 \
        --weight_decay    0.1 \
        --grad_clip       1.0 \
        --min_lr_ratio    0.1 \
        --log_every       20 \
        --val_every       100 \
        --val_steps       20 \
        --checkpoint_every 5000 \
        --checkpoint_dir  /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/nonorm2_lr_${LR} \
        --wandb_project   ece405 \
        --wandb_run_name  "nonorm_lr_${LR}" \
        --ablation        no_norm
done
