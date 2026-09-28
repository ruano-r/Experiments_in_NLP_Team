import re
import sys
import unicodedata

from datasets import load_dataset


# ----- Config --------------------------------------

TARGET = 100_000_000
WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE) # only words, remove punctuation, keep digits
OUTPUT_FILE = "fineweb2_finnish_raw_100m.txt"


# ----- Prepare Data -------------------------------

def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return text.strip()

def tokenize_words(text):

    return WORD_RE.findall(text)

def quality_check(doc):

     language_score = doc.get("language_score", 1.0)
     if language_score < 0.98:
          return False
     return True

# ----- Collect Corpus -----------------------------

def collect_corpus(dataset):

    seen = set()
    total = 0

    documents = []

    for doc in dataset:

        if total >= TARGET:
            break

        if not quality_check(doc):
            continue

        # Simple duplicate detection
        doc_hash = hash(doc["text"][:500])

        if doc_hash in seen:
            continue

        seen.add(doc_hash)

        text = normalize(doc["text"])

        if not text:
            continue

        # Count tokens using the same word tokenizer
        words = tokenize_words(text)
        token_count = len(words)

        if token_count == 0:
            continue

        documents.append(text)

        total += token_count

        if total % 1_000_000 < token_count:
            print(
                f"Collected approximately "
                f"{total:,} / {TARGET:,} tokens"
            )

    return documents, total

# ----- Save to Disk -------------------------------

def write_output(filepath, documents):

    with open(filepath, "w",encoding="utf-8") as f:

        for document in documents:

            # Write the document exactly as collected.
            # No sentence segmentation or reconstruction.
            f.write(document)

            # Separate FineWeb documents with a newline.
            f.write("\n")

# ----- Main ---------------------------------------

def main():

    print("Loading FineWeb-2 Finnish")

    dataset = load_dataset("HuggingFaceFW/fineweb-2", name="fin_Latn", split="train", streaming=True)

    documents, _ = collect_corpus(dataset)

    print("Writing corpus")

    write_output(OUTPUT_FILE,documents)

    print("Corpus is finished")


if __name__ == "__main__":
    main()