'''
Inference entry script using greedy decoding.

Author: Mingbao Lin
References:
- https://huggingface.co/docs/transformers/main/en/generation_strategies
'''

from models.simple_rnn import LM
from inference.encode import greedy_decode
from transformers import AutoTokenizer
import torch


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = LM(tokenizer.vocab_size, hidden_dim= 128, key_dim = 32, value_dim = 32, output_dim = 64, num_layers = 2)
    print(greedy_decode(model, tokenizer, "The sky is", max_length = 20))