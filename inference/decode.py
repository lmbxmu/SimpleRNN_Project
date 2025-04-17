'''
Greedy decoding for sequence generation.

Author: Mingbao Lin
References:
- https://huggingface.co/docs/transformers/main/en/generation_strategies
'''

import torch

def greedy_decode(model, tokenizer, prompt: str, max_length: int = 50, device = "cuda"):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors = "pt")["input_ids"].to(device)
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim = -1, keepdim = True)
            input_ids = torch.cat([input_ids, next_token], dim = 1)
    return tokenizer.decode(input_ids[0], skip_special_tokens = True)


if __name__ == "__main__":
    from ..models.simple_rnn import LM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = LM(tokenizer.vocab_size, hidden_dim= 128, key_dim = 32, value_dim = 32, output_dim = 64, num_layers = 2)
    print(greedy_decode(model, tokenizer, "hello", max_length = 0, device = "cpu"))