import sys
import re
import json
import unicodedata
from collections import Counter


#   ----- Loading trained tokenizer ------------------
def load_merges(path):
    """
    Loading merges from fileon disk
    """
    merges = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            a, b = line.rstrip('\n').split(' ')
            merges.append((a, b))
    return merges

#   ----- Pretokenize --------------------------------

PRETOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)    # catches word characters and punctuation

def pretokenize(line):
    return PRETOKEN_RE.findall(line.lower())

def normalize(text):
    return unicodedata.normalize("NFC", text)

#   ----- BPE apply function --------------------------

def bpe_encode_word(word, merge_ranks, cache=None):
    """
    Based on apply_bpe() from geeks for geeks    
    """

    if cache is not None and word in cache:     # if word has been encoded earlier, return the stored result
        return cache[word]

    symbols = list(word)

    while len(symbols) > 1:
        best_pair = None
        best_rank = float('inf')
        best_idx = None
        
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            rank = merge_ranks.get(pair)        #look up rank (lower rank = learned earlier = higher priority)
            if rank is not None and rank < best_rank:
                best_pair = pair
                best_rank = rank
                best_idx = i

        if best_pair is None:
            break       # no more possible merges

        a, b = best_pair
        symbols = symbols[:best_idx] + [a + b] + symbols[best_idx + 2:]
        # loop again because merging possible created new adjacent pair

    if cache is not None:           
        cache[word] = symbols   # store result so furture occurences of this exact word can be looked up

    return symbols

#   ----- Full corpus encoder -----------------------------------

def encode_line(line, merge_ranks, cache):
    """
    BPE encode every word in a line
    """
    tokens = []
    for word in pretokenize(line):
        pieces = bpe_encode_word(word, merge_ranks, cache)
        tokens.extend(pieces)
    return tokens

def encode_corpus(lines, merge_ranks):
    """
    Creates a shared cache
    Calls encode_line on every line in lines
    Collects each line's token in a list, returns a list of lists
    """
    cache = {}      # same surface word -> same BPE split
    all_tokens = []
    for line in lines:
        all_tokens.append(encode_line(line, merge_ranks, cache))
    return all_tokens  # list of token-lists, one per line

#   ----- Map tokens to id for training LM -------------------------------------------

def tokens_to_ids(all_tokens, string_to_index, unk='<unk>'):
    """
    Converts the string-token corpus to a integer-id corpus
    """
    unk_id = string_to_index[unk]

    # maintains the same structure of all_tokens, but all tokens get their integer-id
    integer_id_corpus =[[string_to_index.get(tok, unk_id) for tok in line] for line in all_tokens]

    return integer_id_corpus


#   ----- Save token ids for the LM training pipeline ---------------------------------

def save_token_ids(id_lines, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in id_lines:
            f.write(' '.join(map(str, line)) + '\n')


#   ---- Main function -----------------------------------------------------------------

def main(saved_bpe, saved_vocab, raw_corpus, token_id_output):

    merges = load_merges(saved_bpe)
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    with open(saved_vocab, encoding='utf-8') as f:
        string_to_index = json.load(f)

    with open(raw_corpus, encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    all_tokens = encode_corpus(raw_lines, merge_ranks)
    id_lines = tokens_to_ids(all_tokens, string_to_index)

    save_token_ids(id_lines, token_id_output) # txt file

    print(f'{len(raw_lines)} lines encoded, vocab size {len(string_to_index)}')

if __name__ == "__main__":
    main( 
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],    
        sys.argv[4]    
    )