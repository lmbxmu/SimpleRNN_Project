# SimpleRNN Language Model

This projects implements a custom recurrent neural network (`SimpleRNN`) and a language modeling framework using PyTorch. It includes training and inference scripts, dataset loading, and basic token generation.

---

## Project Structure

```
simple_rnn_project/
|-- models/                # RNN and LM model definitions
|-- training/              # Training loop
|-- inference/             # Greedy decoding logic
|-- data/                  # Dataset and collate function
|-- run_train.py           # Training entry script (argparse-based)
|-- run_infer.py           # Inference entry script (argparse-based)
|-- train.sh               # Example shell script for training
|-- infer.sh               # Example shell script for inference 
```


## Quick Start

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```


### It is still under development.