# for passing arguments from the command line directly
import sys
import os


import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW


# hyperparameters to select from outside the file, file related, and model related
DATA_FILE = sys.argv[1]
VOCAB_FILE = sys.argv[2]
SAVE_DIR = sys.argv[3]
BLOCK_SIZE = int(sys.argv[4])
BATCH_SIZE = int(sys.argv[5])
LEARNING_RATE = float(sys.argv[6])
DEVICE = sys.argv[7]
EPOCHS = int(sys.argv[8])

"""
for example:
python train.py \
    /scratch/users/john/hungarian_token_ids.txt \
    /scratch/users/john/vocab.json \
    /scratch/users/john/hungarian_model \
    256 \
    32 \
    0.0003 \
    cuda:0 \
    3

"""



#get vocabulary size from vocab.json file

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)


config = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_embd=128,
    n_layer=4,
    n_head=4,
    n_positions=BLOCK_SIZE
)

model = GPT2LMHeadModel(config).to(DEVICE)


# getting the dataser that already contains token ids

class TextDataset(Dataset):

    def __init__(self, file_path, block_size):

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Please create a '{file_path}' file to run this."
            )

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.input_ids = torch.tensor(
            [int(x) for x in text.split()],
            dtype=torch.long
        )

        self.block_size = block_size

    def __len__(self):
        return len(self.input_ids) - self.block_size

    def __getitem__(self, idx):

        chunk = self.input_ids[
            idx:idx + self.block_size
        ]

        return chunk, chunk.clone()


dataset = TextDataset(
    DATA_FILE,
    BLOCK_SIZE
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# training model

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

model.train()

for epoch in range(EPOCHS):

    for batch_idx, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            inputs,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()


# saving to directory

os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)