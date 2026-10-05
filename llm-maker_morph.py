# for passing arguments from the command line directly
import sys
import os
import torch
import numpy as np  
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import AutoTokenizer # for loading
from torch.optim import AdamW


# hyperparameters to select from outside the file, file related, and model related
TOKENIZER = sys.argv[2]
RAW_CORPUS = sys.argv[3]
SAVE_DIR = sys.argv[4]
DEVICE = sys.argv[1]
BLOCK_SIZE = 64
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 3
CHUNK_CHARS = 1_000_000


class TextDataset(Dataset):

    def __init__(self, tokenizer, file_path, block_size):
        # Read the textdata
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Please create a '{file_path}' file to run this."
            )

        self.tokenizer = tokenizer
        self.block_size = block_size

        pieces = []
        total = 0

        def encode(text):
            nonlocal total
            ids = tokenizer(text, add_special_tokens = False)["input_ids"]
            pieces.append(np.array(ids,dtype=np.int32))
            total += len(ids)
            print(f"tokenized {total:,} tokens", flush=True)

        leftover = ""
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                data = f.read(CHUNK_CHARS) #only read the next 1 million characters, file object remembers its position
                if not data:
                    break
                text = leftover + data

                # cut at the last newline or space so words are never split
                cut = max(text.rfind("\n"), text.rfind(" "))
                if cut <= 0:
                    leftover = text
                    continue
                leftover = text[cut:]
                encode(text[:cut])

        if leftover.strip():
            encode(leftover)

        self.input_ids = torch.from_numpy(np.concatenate(pieces))
        print(f"Total tokens: {len(self.input_ids):,}", flush=True)

    def __len__(self):
        return len(self.input_ids) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.input_ids[start:start + self.block_size].long()

        return chunk, chunk.clone()

# Setting Model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, model_max_length=BLOCK_SIZE)

print(tokenizer.mask_token, tokenizer.mask_token_id)
print(tokenizer.pad_token, tokenizer.pad_token_id)
print(len(tokenizer))

config = GPT2Config(vocab_size=tokenizer.vocab_size,
                    n_embd=128,
                    n_layer=4,
                    n_head=4,
                    n_positions=BLOCK_SIZE,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
model = GPT2LMHeadModel(config).to(DEVICE)

dataset = TextDataset(tokenizer, RAW_CORPUS, BLOCK_SIZE) #CHANGE TEXT
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training model

model.train()

for epoch in range(EPOCHS):

    for batch_idx, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs,labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            print(f"epoch {epoch} step {batch_idx} loss {loss.item():.4f}")

# saving to directory

os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
