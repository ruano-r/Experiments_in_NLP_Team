# preprocessed .txt file from the test split of fineweb that we can get from giulias code stats are for the whole file
# tokenizer_dir its the output of wrap_into_HFtok.py 
import os, json, argparse
from collections import Counter
from transformers import AutoTokenizer

BATCH = 1000


def evaluate_dataset(dataset, tokenizer, max_lines=None):
    unk_id = tokenizer.unk_token_id
    vocab_size = len(tokenizer)
    bt = tokenizer.backend_tokenizer

    n_docs = n_words = n_chars = n_tokens = n_unk_tokens = n_unk_words = 0
    token_counts = Counter()

    def process(texts):
        nonlocal n_words, n_chars, n_tokens, n_unk_tokens, n_unk_words
        # add_special_tokens=False to not have the special tokens inflate every count
        encs = bt.encode_batch(texts, add_special_tokens=False)
        for text, enc in zip(texts, encs):
            ids = enc.ids
            words = text.split()           
            # whitespace words for comparing better across tokenizers
            n_words += len(words)
            n_chars += sum(len(w) for w in words)
            n_tokens += len(ids)
            token_counts.update(ids)
            n_unk_tokens += sum(1 for i in ids if i == unk_id)
            # unk words count uses the id 
            n_unk_words += len({w for i, w in zip(ids, enc.word_ids) if i == unk_id})

    buf = []
    for ex in dataset:
        text = ex["text"]
        if not text:
            continue
        n_docs += 1
        if max_lines is not None and n_docs > max_lines:
            n_docs -= 1
            break
        buf.append(text)
        if len(buf) == BATCH:
            process(buf); buf = []
            print(f"{n_docs:,} docs | {n_words:,} words | {n_tokens:,} tokens", flush=True)
    if buf:
        process(buf)

    # simple calculated metric 
    vocab = tokenizer.get_vocab()   
    id_to_tok = {i: t for t, i in vocab.items()}
    covered_types = sum(1 for i in token_counts if id_to_tok.get(i) in vocab)
    covered_occ = sum(c for i, c in token_counts.items() if id_to_tok.get(i) in vocab)
    return {
        "lines": n_docs,
        "words": n_words,
        "characters": n_chars,
        "tokens": n_tokens,
        "fertility_tokens_per_word": n_tokens / n_words,
        "compression_characters_per_token": n_chars / n_tokens,
        "unk_token_rate": n_unk_tokens / n_tokens,
        "unk_word_rate": n_unk_words / n_words,
        "vocabulary_size": vocab_size,
        "observed_token_types": len(token_counts),
        "vocabulary_type_coverage": covered_types / len(token_counts) if token_counts else 0.0,
        "vocabulary_token_coverage": covered_occ / n_tokens,
    }


def print_results(r):
    print(f"Lines: {r['lines']:,} | Words: {r['words']:,} | "
          f"Chars: {r['characters']:,} | Tokens: {r['tokens']:,}")
    print(f"Fertility:   {r['fertility_tokens_per_word']:.4f} tokens/word")
    print(f"Compression: {r['compression_characters_per_token']:.4f} chars/token")
    print(f"<unk> token rate: {r['unk_token_rate']*100:.4f}%")
    print(f"<unk> word rate:  {r['unk_word_rate']*100:.4f}%")
    print(f"Vocabulary size: {r['vocabulary_size']:,}")
    print(f"Observed token types: {r['observed_token_types']:,}")
    print(f"Vocabulary type coverage:  {r['vocabulary_type_coverage']*100:.4f}%")
    print(f"Vocabulary token coverage: {r['vocabulary_token_coverage']*100:.4f}%")



def read_lines(path):
    """ stats are aggregated over the whole file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield {"text": line}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer_dir")
    p.add_argument("text_file")
    p.add_argument("max_lines", nargs="?", type=int, default=None)
    p.add_argument("--name", default=None, help="used in the output filename")
    a = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    print(f"Loaded tokenizer, vocab size {len(tokenizer):,}")

    results = evaluate_dataset(read_lines(a.text_file), tokenizer, a.max_lines)
    print_results(results)

    name = a.name or os.path.splitext(os.path.basename(a.text_file))[0]
    out = f"tokenizer_eval_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to: {out}")


if __name__ == "__main__":
    main()