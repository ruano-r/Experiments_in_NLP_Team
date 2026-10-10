# This script evaluates a model on a dev and/or test set.
# It gives also perplexity and bit measures, to be comparable across different tokenizers
#call: python eval_model MODEL_DIR TOK_DIR DEV/TEST_FILE OUT_FILE

import os
import math
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel
import sys

MODEL_DIR = sys.argv[1]
TOK_DIR = sys.argv[2]
IMPORT_FILE = sys.argv[3]
OUT_FILE = sys.argv[4]

BLOCK_SIZE = 64
EVAL_BATCH_SIZE = 64
CHUNK_CHARS = 1_000_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#same word definition as preprocessing
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# ---- word/text stats
def count_words(text):
    return len(WORD_RE.findall(text))


def get_text_statistics(file_path):
    """
    Count lines, words and characters in the raw dev file.
    """

    n_lines = 0
    n_words = 0
    n_chars = 0
    n_bytes = 0 #we take bytes for later
    

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            text = f.read(CHUNK_CHARS)

            if not text:
                break

            n_lines += text.count("\n")
            n_words += count_words(text)
            n_chars += len(text)
            n_bytes += len(text.encode("utf-8"))
            

    return {
        "lines": n_lines,
        "words": n_words,
        "chars": n_chars,
        
        "bytes": n_bytes,
        }

# same procedure as training
class ImportedDataset(Dataset):

    def __init__(self, tokenizer, file_path, block_size):

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        self.block_size = block_size

        pieces = []
        total_tokens = 0
        
        def encode(text):
          nonlocal total_tokens
          ids = tokenizer(text, add_special_tokens=False)["input_ids"]
          pieces.append(np.array(ids, dtype=np.int32))
          total_tokens += len(ids)
          print(f"tokenized {total_tokens:,} tokens", flush=True)

        leftover = ""
        with open(file_path, "r", encoding="utf-8") as f:
          while True:
            data = f.read(CHUNK_CHARS)
            if not data:
              break
            text = leftover + data
            
            cut = max(text.rfind("\n"), text.rfind(" "))
            if cut <= 0:
              leftover = text
              continue
            leftover = text[cut:]
            encode(text[:cut])

        
        if leftover.strip():
            encode(leftover)

        self.input_ids = torch.from_numpy(np.concatenate(pieces))

        self.total_tokens = len(self.input_ids)

    def __len__(self):
      return self.total_tokens // self.block_size

    def __getitem__(self, idx):
      start = idx * self.block_size
      chunk = self.input_ids[
            start:start + self.block_size].long()

      return chunk


#---- load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR, model_max_length=BLOCK_SIZE)

print("Tokenizer loaded!")
print(f"Tokenizer vocab: {len(tokenizer):,}")


#---- load model

model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)

model.eval()

n_parameters = sum(p.numel() for p in model.parameters())

print("Model loaded!")
print(f"Model parameters: {n_parameters} ")

#---- text statistics
text_stats = get_text_statistics(IMPORT_FILE)

n_lines = text_stats["lines"]
n_words = text_stats["words"]
n_chars = text_stats["chars"]
n_bytes = text_stats["bytes"]

#----tokenize import file
dataset = ImportedDataset(
    tokenizer,
    IMPORT_FILE,
    BLOCK_SIZE
)

dataloader = DataLoader(
    dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False
)

n_tokens = dataset.total_tokens
n_blocks = len(dataset)

#tokens fed to the model (includes the unscored first token of each block)
n_eval_tokens = n_blocks * BLOCK_SIZE 

n_batches = len(dataloader)

#----unk token
unk_id = tokenizer.unk_token_id
unk_count = int((dataset.input_ids == unk_id).sum().item()) if unk_id is not None else 0
unk_rate = unk_count / n_tokens

#---- model eval
print(
    f"Blocks: {n_blocks:,} | "
    f"Batches: {n_batches:,} | "
    f"Batch size: {EVAL_BATCH_SIZE}")

print()

total_loss = 0.0
total_eval_tokens = 0
total_correct = 0 #for top1 accuracy

with torch.no_grad():

    for batch_idx, x in enumerate(dataloader):

        x = x.to(DEVICE)

        outputs = model(x, labels=x)

        loss = outputs.loss
        batch_tokens = x.size(0) * (x.size(1) - 1)

        preds = outputs.logits[:, :-1].argmax(dim=-1)
        total_correct += (preds == x[:, 1:]).sum().item()

        total_loss += (loss.item() * batch_tokens)

        total_eval_tokens += batch_tokens

        print(
            f"{IMPORT_FILE} batch "
            f"{batch_idx + 1}/{n_batches} "
            f"loss {loss.item():.4f}",
            flush=True)

#---- final loss
import_loss= (total_loss / total_eval_tokens)

next_token_accuracy = total_correct / total_eval_tokens

#for calculating and understanding model metrics i used:
#https://stackoverflow.com/questions/61988776/how-to-calculate-perplexity-for-a-language-model-using-pytorch
#https://medium.com/@shubhamsd100/understanding-perplexity-in-language-models-a-detailed-exploration-2108b6ab85af$0
#https://stackoverflow.com/questions/71089293/inconsistency-in-interpreting-python-float-in-pytorch-why

#---- model metrics 

#negative log likelihood (its in nats!)
total_nll = total_loss

# share of tokens actually scored: the first token of each block gets
#no context so it can never be a target + incomplete final block is discarded
# in this way we are assuming that unscored tokens have avg length
coverage = total_eval_tokens / n_tokens

#perplexity per token
#torch.exp expect a tensor float while loss is python float > convert
ppl_per_token = torch.exp(torch.tensor(import_loss, dtype=torch.float64)).item()

# perplexity per words
ppl_per_word = torch.exp(torch.tensor(total_nll / (n_words * coverage), dtype=torch.float64)).item()

# nll to bits
total_bits = total_nll / math.log(2) #convert nats to bits

# bits per character
bits_per_character = total_bits / (n_chars * coverage)

# bits per byte (UTF-8) (tokenizer comparable!)
bits_per_byte = total_bits / (n_bytes * coverage)

#how much "text" one token has on average (comparable for tokenizers)
bytes_per_token = n_bytes / n_tokens if n_tokens > 0 else float("nan")

#how many bits of info the model needs to say which token comes next (on avg) 
# (not comparable across toks > how hard the prediction is depends also on how much text it has seen > the bytes token)
bits_per_token = total_bits / total_eval_tokens


lines = []
lines.append("Model evals")
lines.append("#### Model & Tok")
lines.append(f"Model:           {MODEL_DIR}")
lines.append(f"Tokenizer:       {TOK_DIR}")
lines.append(f"Evaluated split: {IMPORT_FILE}")
lines.append(f"Parameters:      {n_parameters:,}")
lines.append("")
lines.append("#### Text Stats")
lines.append(f"Lines:           {n_lines:,}")
lines.append(f"Words:           {n_words:,}")
lines.append(f"Characters:      {n_chars:,}")
lines.append(f"Bytes (UTF-8):   {n_bytes:,}")
lines.append("")
lines.append("#### Token level infos")
lines.append(f"Tok tokens:      {n_tokens:,}")
lines.append(f"Bytes/token:     {bytes_per_token:.4f}")
lines.append(f"Tokens scored:   {total_eval_tokens:,}")
lines.append(f"Block size:      {BLOCK_SIZE}")
lines.append(f"Token coverage:  {coverage:.4%}")
lines.append(f"UNK tokens:      {unk_count:,} ({unk_rate:.6%})")

lines.append("")
lines.append("#### Model metrics")
lines.append(f"Mean Loss on import file: {import_loss:.6f}")
lines.append(f"Total NLL (in nats):      {total_nll:.2f}")
lines.append(f"Perplexity/token:         {ppl_per_token:.4f}")
lines.append(f"Perplexity/word:          {ppl_per_word:.4f}")
lines.append(f"Bits/token:               {bits_per_token:.6f}")
lines.append(f"Bits/character:           {bits_per_character:.6f}")
lines.append(f"Bits/byte:                {bits_per_byte:.6f}")
lines.append(f"Next-token accuracy:      {next_token_accuracy:.4%}")

summary = "\n".join(lines)
print(summary)

with open(OUT_FILE, 'w', encoding="utf-8") as f:
   f.write(summary + "\n")

print(f"\nResults written to {OUT_FILE}")


