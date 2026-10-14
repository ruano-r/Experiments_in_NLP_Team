import os, json, string
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, trainers
import sys

# python BPE_baseline_tokenizer.py SEGMENTED_TRAINFILE OUT_DIR


SEGMENTED = sys.argv[1]     # the usual training 1M @@-segmented file
OUT_DIR = sys.argv[2]

target_size = 20000

def words():
    result = []
    with open(SEGMENTED, encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                result.append(w.replace("@@", ""))     # same words, no segmentation
    return result

tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
tok.pre_tokenizer = pre_tokenizers.Whitespace()

#hardcode YOUR LANGUAGE list of basic items
#THIS LIST HAS TO MATCH EXACTLY WHAT WAS DONE FOR THE OTHER TOKENIZERS!!!
base = list(string.punctuation + "¡¿«»“”‘’–—…€°ºª" + string.digits
            + "abcdefghijklmnopqrstuvwxyzáéíóúüñç")
trainer = trainers.BpeTrainer(
    vocab_size=target_size,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    initial_alphabet=base,
)
tok.train_from_iterator(words(), trainer=trainer)

os.makedirs(OUT_DIR, exist_ok=True)
tok.model.save(OUT_DIR)            # writes vocab.json and merges.txt