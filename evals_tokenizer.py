import sys
import json
from collections import Counter
from datasets import load_dataset

#to import the tokenizer code
from tokenize_data import (
    load_merges,
    pretokenize,
    bpe_encode_word,
)

#open json
def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)



def evaluate_dataset(
    dataset,
    merge_ranks,
    vocab,
    max_documents=None,
):
    """
    Evaluate our BPE tokenizers on a hggingface dataset on the following metrics

        - fertility (tokens per word)
        - compression (characters per token)
        - <unk> rate (rate of unknown tokens)
        - vocabulary coverage (how much vocab is generalized enough to also be on our test) -- for types and for tokens too
    """

    #1 getting the counts
    #2 calculating the metrics
    
    #1.

    #caching like in the tokenizer
    cache = {}

    #counts
    num_documents = 0
    num_words = 0
    num_chars = 0
    num_tokens = 0

    num_unk_tokens = 0
    num_unk_words = 0

    #tokens and types
    observed_token_types = set()
    observed_token_counts = Counter()

    for example in dataset:
        text = example["text"]

        if not text:
            continue

        num_documents += 1

        if max_documents is not None and num_documents > max_documents:
            break

        #using existing preprocessing
        words = pretokenize(text)

        for word in words:
            if not word:
                continue
            
            #count words and characters
            num_words += 1
            num_chars += len(word)

            #using existing bpe implementation
            tokens = bpe_encode_word(
                word,
                merge_ranks,
                cache,
            )
            #count tokens 
            num_tokens += len(tokens)
            #not an unknown
            word_has_unk = False

            for token in tokens:
                #adding the types and counting them
                observed_token_types.add(token)
                observed_token_counts[token] += 1
                
                if token not in vocab:
                    num_unk_tokens += 1
                    word_has_unk = True

            if word_has_unk:
                num_unk_words += 1

        if num_documents % 1000 == 0:
            print(
                f"{num_documents:,} documents | "
                f"{num_words:,} words | "
                f"{num_tokens:,} tokens"
            )





    # 2. 

    # Fertility
    fertility = num_tokens / num_words

    # Compression 
    compression = num_chars / num_tokens

    # UNK token rate
    unk_token_rate = num_unk_tokens / num_tokens

    # UNK word rate
    unk_word_rate = num_unk_words / num_words

    # Vocabulary coverage
 
    # for types
    covered_types = sum(
        1 for token in observed_token_types
        if token in vocab
    )

    type_coverage = (
        covered_types / len(observed_token_types)
        if observed_token_types
        else 0.0
    )

    # for tokens, based on occurences
    covered_token_occurrences = sum(
        count
        for token, count in observed_token_counts.items()
        if token in vocab
    )

    token_coverage = covered_token_occurrences / num_tokens

    return {
        "documents": num_documents,
        "words": num_words,
        "characters": num_chars,
        "tokens": num_tokens,

        "fertility_tokens_per_word": fertility,

        "compression_characters_per_token": compression,

        "unk_token_rate": unk_token_rate,
        "unk_word_rate": unk_word_rate,

        "vocabulary_size": len(vocab),

        "observed_token_types": len(observed_token_types),

        "vocabulary_type_coverage": type_coverage,
        "vocabulary_token_coverage": token_coverage,
    }


#printing the results to be easier to read

def print_results(results):

    print()
    print("=" * 60)
    print("TOKENIZER EVALUATION")
    print("=" * 60)

    print(f"Documents:              {results['documents']:,}")
    print(f"Words:                  {results['words']:,}")
    print(f"Characters:             {results['characters']:,}")
    print(f"Tokens:                 {results['tokens']:,}")
    print()

    print(
        f"Fertility:              "
        f"{results['fertility_tokens_per_word']:.4f} tokens/word"
    )

    print(
        f"Compression:            "
        f"{results['compression_characters_per_token']:.4f} chars/token"
    )

    print()

    print(
        f"<unk> token rate:       "
        f"{results['unk_token_rate'] * 100:.4f}%"
    )

    print(
        f"<unk> word rate:        "
        f"{results['unk_word_rate'] * 100:.4f}%"
    )

    print()

    print(
        f"Vocabulary size:        "
        f"{results['vocabulary_size']:,}"
    )

    print(
        f"Observed token types:   "
        f"{results['observed_token_types']:,}"
    )

    print(
        f"Vocabulary type coverage:"
        f" {results['vocabulary_type_coverage'] * 100:.4f}%"
    )

    print(
        f"Vocabulary token coverage:"
        f" {results['vocabulary_token_coverage'] * 100:.4f}%"
    )

    print("=" * 60)





#you can define the things you want from the terminal, in the order preovided in the print statement

def main():

    if len(sys.argv) < 5:
        print(
            "Usage:\n"
            "python evaluate_tokenizer.py "
            "<merges> <vocab> <language>(specify as stated in finewb) <split>(test or dev) "
            "[max_documents]"
        )
        sys.exit(1)

    merges_path = sys.argv[1]
    vocab_path = sys.argv[2]
    language = sys.argv[3]
    split = sys.argv[4]

    max_documents = None

    if len(sys.argv) >= 6:
        max_documents = int(sys.argv[5])

  
    #loading the tokenizer
    merges = load_merges(merges_path)

    merge_ranks = {
        pair: rank
        for rank, pair in enumerate(merges)
    }

    print(f"Loaded {len(merges):,} merges.")

    vocab = load_vocab(vocab_path)
    print(f"Loaded vocabulary with {len(vocab):,} entries.")



    # loading fineweb

    print("Loading FineWeb-2...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        language,
        split=split,
        streaming=True,
    )

    
    
    # getting the straightforward stats 
    
    results = evaluate_dataset(
        dataset,
        merge_ranks,
        vocab,
        max_documents=max_documents,
    )

    print_results(results)

    # saving the results
    output_path = (
        f"tokenizer_eval_{language}_{split}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()