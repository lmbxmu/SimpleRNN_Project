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


def train_model(model, dataloader, num_epochs, lr, device = "cuda", use_amp = False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scaler = torch.amp.GradScaler("cuda", enabled = use_amp) # relieve underflow issue from low-precision training
    loss_fn = nn.CrossEntropyLoss(reduction = "none")   # we need to apply a mask to remove padding tokens from loss calculation

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_ids, labels, seq_lens in tqdm(dataloader, desc = f"Epoch {epoch + 1}"):
            input_ids, labels, seq_lens = input_ids.to(device), labels.to(device), seq_lens.to(device)
            optimizer.zero_grad()

            # Align input_ids and labels for next-token prediction
            input_ids = input_ids[:, :-1] # remove the last token (eos) from input_ids
            labels = labels[:, 1:]  # remove the first token (bos) from labels --> right shift for one token
            seq_lens -= 1 # remove the last token (eos) from seq_lens

            with torch.amp.autocast("cuda", enabled = use_amp): # support mixed-precision training (fp16 or bf16)
                logits = model(input_ids)
                B, T, V = logits.shape

                # Reshape logits and labels for CrossEntropyLoss
                logits_flat = logits.reshape(-1, V) # [B*T, V]
                labels_flat = labels.reshape(-1) # [B*T]
                # Calculate loss per token
                loss_per_token = loss_fn(logits_flat, labels_flat) # [B*T]

                # Generate mask: [B, T]
                mask = torch.arange(T, device = device).expand(B, T) < seq_lens.unsqueeze(1) # [B, T]
                mask = mask.float().view(-1)  # [B*T]
                
                # Apply mask to loss per token
                loss = (loss_per_token * mask).sum() / mask.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}")



if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from models.simple_rnn import LM
    from data.dataset import CharDataset, collate_batch
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer")
    text = ["Hello world! This is a test to verify the correctness of dataloader.", \
            "Hellow, this is a test, number 2.", "Hellow", "nice to meet you!",  \
            "My great honor to meet you!", "How are you doing today?"]
    dataset = CharDataset(text, tokenizer)
    dataloader = DataLoader(dataset, batch_size = 2, collate_fn = collate_batch)

    model = LM(tokenizer.vocab_size + 1, hidden_dim= 128, key_dim = 128, value_dim = 128, output_dim = 128, num_layers = 2)
    train_model(model, dataloader, num_epochs = 20, lr = 1e-3, device = "cpu")