'''
Training entry script for LM with SimpleRNN.

Author: Mingbao Lin
- https://huggingface.co/docs/transformers/tasks/language_modeling
'''


from models.simple_rnn import LM
from training.train import train_model
from data.dataset import CharDataset, collate_batch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2") 
    dataset = CharDataset("Hello world! This is a test to verify the correctness of dataloader.", tokenizer, seq_len = 8)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, collate_fn = collate_batch)
    model = LM(tokenizer.vocab_size, hidden_dim= 128, key_dim = 32, value_dim = 32, output_dim = 64, num_layers = 2)
    train_model(model, dataloader, num_epochs = 5, lr = 1e-3)