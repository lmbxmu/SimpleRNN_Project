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