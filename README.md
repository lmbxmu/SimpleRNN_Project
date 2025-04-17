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

```
pip3 install -r requirements.txt
```

### 2. Train the model

```bash
bash train.sh
```

or directly:
```bash
python3 run_train.py \
   --text "Hello world! This is a test to verify the correctness of dataloader." \
   --seq_len 8 \
   --batch_size 4 \
   --num_epochs 5 \
   --lr 0.001 \
   --device gpu
```

---


### 2. Run inference

```bash
bash infer.sh
```

or directly:

```bash
python3 run_infer.py --prompt "The sky is" --max_length 20 --device cpu
```

---

## References

- [PyTorch RNN Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers)  
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

---


## Author
- Mingbao Lin


### It is still under development.