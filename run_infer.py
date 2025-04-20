'''
Inference entry script using greedy decoding.

Author: Mingbao Lin
References:
- https://huggingface.co/docs/transformers/main/en/generation_strategies
'''

from models.simple_rnn import LM
from inference.decode import greedy_decode
from transformers import AutoTokenizer
import torch

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run inference with SimpleRNN-based LM")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer")
    model = LM(vocab_size = tokenizer.vocab_size + 1, hidden_dim = 128, key_dim = 128, value_dim = 128, output_dim = 128, num_layers = 2)
    model.to(args.device)

    result = greedy_decode(model, tokenizer, args.prompt, max_length = args.max_length, device = args.device)
    print("Generated text:", result)


if __name__ == "__main__":
    main()