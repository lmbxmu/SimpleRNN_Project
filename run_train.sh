#!/bin/bash
# Example training script

python3 run_train.py \
   --file_path ./test_dataset/training.jsonl \
   --save_every_n_epochs 5 \
   --batch_size 2 \
   --num_epochs 20 \
   --lr 0.001 \
   --device cuda
