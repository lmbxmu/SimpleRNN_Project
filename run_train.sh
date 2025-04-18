#!/bin/bash
# Example training script

python3 run_train.py \
   --text "Hello world! This is a test to verify the correctness of dataloader." \
   --seq_len 8 \
   --batch_size 4 \
   --num_epochs 5 \
   --lr 0.001 \
   --device cuda
