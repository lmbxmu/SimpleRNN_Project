'''
Training loop with AMP support.

Author: Mingbao Lin
References:
- https://pytorch.org/tutorials
- https://github.com/huggingface/transformers
'''


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, dataloader, num_epochs, lr, device = "cuda", use_amp = False)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scaler = torch.cuda.amp.GradScaler(enabled = use_amp)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_ids, labels in tqdm(dataloader, desc = f"Epoch {epoch + 1}"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled = use_amp):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}")