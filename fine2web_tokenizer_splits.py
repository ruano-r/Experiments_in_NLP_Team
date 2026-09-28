import re
import sys
from datasets import load_dataset

#specify the target number of words for your dataset, and the output file name. See example below
#(bash)
#python fine2web_tokenizer_splits.py 1000000 "$TMPDIR/hungarian_1m.txt

#target and output arguments
TARGET = int(sys.argv[1])
OUTPUT = sys.argv[2]

#regex for links and words (no standalone punctuation or numbers)
url_re = re.compile(r"^(https?://|www\.)", re.I)
letter_re = re.compile(r"[^\W\d_]", re.UNICODE)

#is a word and not a link
def is_counted(word):
    return bool(letter_re.search(word)) and not url_re.match(word)


#change the language and split accordingly
dataset = load_dataset(
    "HuggingFaceFW/fineweb-2",
    "hun_Latn",
    split="train",
    streaming=True,
)

total = 0

with open(OUTPUT, "w", encoding="utf-8") as f:
    #keeping doc separation
    for doc in dataset:
        text = doc["text"]
        count = sum(is_counted(word) for word in text.split())

        #until it reaches the target amount
        if total + count > TARGET:
            break

        f.write(text)
        f.write("\n")

        total += count

        if total % 1_000_000 == 0:
            print(f"{total:,} words", flush=True)

#get location and amount of words as confirmation
print(f" {total:,} words")
print(f"Output file: {OUTPUT}")