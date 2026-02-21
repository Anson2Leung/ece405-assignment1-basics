#!/bin/bash
#SBATCH --job-name=ECE405_7.2batchsize
#SBATCH --gres=gpu:1
#SBATCH --account=ece405
#SBATCH --partition=ece405
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --error=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/batch_error%j.log
#SBATCH --output=/home/ansonl32/koa_scratch/ECE405/assignment1-basics/logs/batch%j.log

export WANDB_API_KEY=""

export CUDA_LAUNCH_BLOCKING=1

TOK_TRAIN="/home/ansonl32/ECE405/data/tinystories_train.npy"
TOK_VALID="/home/ansonl32/ECE405/data/tinystories_valid.npy"

# TODO: Remember to switch out LR
BEST_LR=1e-3

cd /home/ansonl32/ECE405/ece405-assignment1-basics

# Keep total tokens fixed at 327,680,000
# max_steps = 327,680,000 / (batch_size * context_length)
# context_length = 256
for BATCH_SIZE in 1 8 16 48; do

    echo "========================================="
    echo "BATCH_SIZE=$BATCH_SIZE"
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
        --batch_size      "$BATCH_SIZE" \
        --max_steps       5000 \
        --warmup_steps    500 \
        --lr              "$BEST_LR" \
        --beta1           0.9 \
        --beta2           0.999 \
        --weight_decay    0.1 \
        --grad_clip       1.0 \
        --min_lr_ratio    0.1 \
        --log_every       100 \
        --val_every       200 \
        --val_steps       20 \
        --checkpoint_every 5000 \
        --checkpoint_dir  /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/TSbatch_${BATCH_SIZE} \
        --wandb_project   ece405 \
        --wandb_run_name  "TSbatch_${BATCH_SIZE}_lr_${BEST_LR}"
done
