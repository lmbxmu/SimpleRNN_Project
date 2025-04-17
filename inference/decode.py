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