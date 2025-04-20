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
    def __init__(self, text: list[str], tokenizer):

        for i in range(len(text)):
            text[i] = tokenizer.bos_token + text[i] + tokenizer.eos_token # add BOS and EOS tokens
        self.input_ids = tokenizer(text, return_tensors = "pt", padding = "longest", truncation = True)["input_ids"]
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):
        input_seq = self.input_ids[idx, :]
        target_seq = self.input_ids[idx, 1:]
        return input_seq, target_seq


def collate_batch(batch):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader


    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer")
    text = ["Hello world! This is a test to verify the correctness of dataloader.", \
            "Hellow, this is a test, number 2.", "Hellow", "nice to meet you!",  \
            "My great honor to meet you!", "How are you doing today?"]

    dataset = CharDataset(text, tokenizer)
    for i in range(len(text)):
        input_seq, target_seq = dataset[i]
        print(f"Input {i}: {input_seq}, Target: {target_seq}")

    
    dataloader = DataLoader(dataset, batch_size = 2, collate_fn = collate_batch)
    for batch in dataloader:
        input_batch, target_batch = batch
        print("Input batch shape:", input_batch.shape)
        print("Target batch shape:", target_batch.shape)
        print("Input batch:", input_batch)
        print("Target batch:", target_batch)
        break