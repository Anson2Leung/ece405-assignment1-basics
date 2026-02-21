# temp file to run experiments 
#nohup bash KoaInteractive.sh &

# remember to insert and remove

export WANDB_API_KEY="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"
export CUDA_LAUNCH_BLOCKING=1

TOK_TRAIN="/home/ansonl32/ECE405/data/tinystories_train.npy"
TOK_VALID="/home/ansonl32/ECE405/data/tinystories_valid.npy"

cd /home/ansonl32/ECE405/ece405-assignment1-basics

for LR in 1e-3 9e-4; do
    for ABLATION in nope; do
        echo "========================================="
        echo "Ablation: $ABLATION | LR=$LR"
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
            --checkpoint_dir  /home/ansonl32/koa_scratch/ECE405/assignment1-basics/checkpoints/nope_${ABLATION} \
            --wandb_project   ece405 \
            --wandb_run_name  "nope_${ABLATION}_$LR" \
            --ablation        "$ABLATION"
    done
done
