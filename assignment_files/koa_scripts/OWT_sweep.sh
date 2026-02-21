#!/bin/bash
#SBATCH --job-name=ECE405_OWT_sweep
#SBATCH --gres=gpu:1
#SBATCH --account=ece405
#SBATCH --partition=ece405
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --error=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/owt_sweep_error%j.log
#SBATCH --output=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/owt_sweep%j.log

export WANDB_API_KEY=""
export CUDA_LAUNCH_BLOCKING=1

TOK_TRAIN="/home/ansonl32/ECE405/data/owt_train.npy"
TOK_VALID="/home/ansonl32/ECE405/data/owt_valid.npy"

cd /home/ansonl32/ECE405/ece405-assignment1-basics

for LR in 3e-3 5e-3 8e-3; do
    echo "========================================="
    echo "Starting OWT sweep run with LR=$LR"
    echo "========================================="
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
        --batch_size      32 \
        --max_steps       3000 \
        --warmup_steps    300 \
        --lr              "$LR" \
        --beta1           0.9 \
        --beta2           0.999 \
        --weight_decay    0.1 \
        --grad_clip       1.0 \
        --min_lr_ratio    0.1 \
        --log_every       50 \
        --val_every       200 \
        --val_steps       20 \
        --checkpoint_every 3000 \
        --checkpoint_dir  /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/owt_sweep2_lr${LR} \
        --wandb_project   ece405 \
        --wandb_run_name  "owt_sweep_lr${LR}" \
        --ablation        none
done