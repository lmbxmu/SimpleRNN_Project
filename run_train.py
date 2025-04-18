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

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train SimpleRNN-based LM")
    parser.add_argument("--text", type=str, required=True, help="Training text content")
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2") 
    dataset = CharDataset(args.text, tokenizer, seq_len = args.seq_len)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, collate_fn = collate_batch)

    model = LM(vocab_size = tokenizer.vocab_size, hidden_dim = 128, key_dim = 128, value_dim = 128, output_dim = 128, num_layers = 2)
    train_model(model, dataloader, num_epochs = args.num_epochs, lr = args.lr, device = args.device)


if __name__ == "__main__":
    main()