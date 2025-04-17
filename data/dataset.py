'''
Character-level dataset for autoregressive language modeling.
Prepares input-target token pairs for next-token prediction


Author: Mingbao Lin
Acknowledgement: Inpsired by tokenization and batching routines in HuggingFace and PyTorch NLP examples.
References:
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
'''


from torch.utils.data import Dataset
import torch

class CharDataset(Dataset):
    def __init__(self, text: str, tokenizer, seq_len: int):
        self.input_ids = tokenizer(text, return_tensors = "pt")["input_ids"][0]
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.input_ids) - self.seq_len
    
    def __getitem__(self, idx):
        input_seq = self.input_ids[idx: idx + self.seq_len]
        target_seq = self.input_ids[idx + 1: idx + 1 + self.seq_len]
        return input_seq, target_seq


def collate_batch(batch):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader


    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = CharDataset("Hello world! This is a test to verify the correctness of dataloader.", tokenizer, seq_len = 8)
    for i in range(3):
        input_seq, target_seq = dataset[i]
        print(f"Input: {input_seq}, Target: {target_seq}")

    
    dataloader = DataLoader(dataset, batch_size = 2, collate_fn = collate_batch)
    for batch in dataloader:
        input_batch, target_batch = batch
        print("Input batch shape:", input_batch.shape)
        print("Target batch shape:", target_batch.shape)
        print("Input batch:", input_batch)
        print("Target batch:", target_batch)
        break