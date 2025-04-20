#!/bin/bash
# Example training script

python3 run_train.py \
   --file_path ./test_dataset/training.jsonl \
   --batch_size 4 \
   --num_epochs 10 \
   --lr 0.001 \
   --device cuda
