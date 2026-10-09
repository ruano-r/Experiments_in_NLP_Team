#   MORPHO-AWARE BPE TOKENIZER
#
#   This tokenizer is an adaptation of the known BPE algorithm and has an additional variable that can be tuned for strictness and cross-boundary morpheme merging. 
#
#   Based on: https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/

import sys
import json
import string
from collections import Counter, defaultdict

#   ----- Structuring data -----------------------

class Word:
    """
    A word is a frequency + a list of morphemes
    Each morpheme is itself a list of characters (that will merge along the way)
    """
    def __init__(self, morphemes, freq):
        self.morphemes = morphemes
        self.freq = freq

def create_word_objects(lines):
    """
    Parse the training data into Word objects

    Data should be morphological segmented and @@-marked, with no spaces between the different morphemes of surface-words: walk@@ed
    """

    word_freq = Counter(lines) # count word appearance frequency in complete segmented dataset
    vocab = []

    for token, freq in word_freq.items():
        morphs = token.split('@@')                  #['walk', 'ed']
        morphemes = [list(m) for m in morphs]       #[['w', 'a', 'l', 'k'],['e', 'd']]
        morphemes[0].insert(0, '_')                 # '_' as marker for the beginning of a word: [['_','w', 'a', 'l', 'k'],['e', 'd']]
        vocab.append(Word(morphemes, freq)) 

    return vocab

def is_complete(morpheme):
    """
    A morpheme is considered 'complete' when it has been merged into one symbol
    """
    return len(morpheme) == 1


#   ----- Pair counting -------------------------------------

def get_pair_frequencies(vocab):
    """
    Counts how often each adjacent pair occurs in the current tokenized vocabulary.
    Intra and inter pairs are counted seperately.
    Inter pairs can only happen if both morphemes are 'complete'
    """

    intra_counts = Counter()
    inter_counts = Counter()
    

    for word in vocab:
        morphemes = word.morphemes

        for m_idx, morph in enumerate(morphemes):       # loop over morphemes in Word

            # intra-morpheme pairs        
            for i in range(len(morph) - 1):
                pair = (morph[i], morph[i+1])
                intra_counts[pair] += word.freq # this pair appears as often as the word appears, if pair is present in word multiple times the frequency will increased in the next iteration where this pair is seen

            # inter-morpheme pairs: cross boundary pair
            # only possible if both morphemes are already 'complete'
            if m_idx + 1 < len(morphemes):      # check if there is a morpheme ofter the current one
                nxt = morphemes[m_idx + 1]
                if is_complete(morph) and is_complete(nxt):
                    pair = (morph[0], nxt[0])
                    inter_counts[pair] += word.freq
            
    return intra_counts, inter_counts


#   ----- Training BPE -------------------------------------------

def learn_bpe(vocab, threshold, vocab_size): 
    """
    This function selects the most frequent merge, adds it to the list of merges and applies the merge.
    """

    merges = []

    for _ in range(vocab_size):
        intra_counts, inter_counts = get_pair_frequencies(vocab)

        if not intra_counts and not inter_counts:
            break

        allowed = {}        # pair --> (score, merge_inter) with merge_inter being a boolean
        for pair in set(intra_counts)| set(inter_counts):
            intra = intra_counts.get(pair, 0) # get frequency of pair, return 0 if there is no value
            inter = inter_counts.get(pair, 0)

            inter_allowed = inter > threshold #is the frequency of inter pair above threshold

            if inter_allowed:
                score = intra + inter
            else: 
                score = intra

            if score > 0:
                allowed[pair] = (score, inter_allowed)
      
        if not allowed:     #if no allowed pairs left, break
            break

        most_frequent = max(allowed, key=lambda p: allowed[p][0]) # get most frequent merge
        merge_inter = allowed[most_frequent][1] # check if merging inter is allowed

        merges.append(most_frequent)
        apply_merge(vocab, most_frequent, merge_inter)

    return merges


#   ----- Merging pairs ---------------------------------------------

def apply_merge(vocab, pair, merge_inter):
    """
    FUNCTION DESCRIPTION
    """

    a, b = pair
    merged_symbol = a + b

    for word in vocab:

        # merge within each morpheme
        for i, morph in enumerate(word.morphemes):
            word.morphemes[i] = merge_within(morph, a, b, merged_symbol)


        # merge two adjacent 'complete' morphemes
        if merge_inter:
            word.morphemes = merge_across(word.morphemes, a, b, merged_symbol)


def merge_within(morph, a, b, merged):
    """
    This function replaces every occurrence of a selected pair with a single merged token within morpheme boundaries
    """
    out, i = [], 0

    while i < len(morph):
        if i < len(morph) - 1 and morph[i] == a and morph[i+1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(morph[i])
            i += 1

    return out

def merge_across(morphs, a, b, merged):
    """
    This function merges adjacent complete morphmes into one single-symbol morpheme
    """
    out, i = [], 0

    while i < len(morphs):
        if (i + 1 < len(morphs)
                and is_complete(morphs[i]) and morphs[i][0] == a
                and is_complete(morphs[i+1]) and morphs[i+1][0] == b):
            out.append([merged])            # the two morphemes are now one unit, so one list item
            i += 2
        else:
            out.append(morphs[i])
            i += 1

    return out

def build_vocab_from_merges(merges, chars, specials=('<unk>', '<pad>', '<bos>', '<eos>')):
    """
    Builds the complete vocabulary from the merges.
    Both parts of the merge are added to the symbols as well as the combined symbol.
    The index-to-string and string-to-index mappings are created from this list of symbols.
    """    

    symbols = set(chars) # all characters 
    punctuation = [p for p in string.punctuation]
    digits = [d for d in string.digits]

    merges = list(dict.fromkeys(merges)) #remove duplicate merges from list
    
    for a, b in merges:
        symbols.add(a)
        symbols.add(b)
        symbols.add(a + b)
    index_to_string = list(specials) + punctuation + digits + sorted(symbols) # just a list, but the index of each item is their id
    string_to_index = {tok: i for i, tok in enumerate(index_to_string)}

    return string_to_index, index_to_string

#   ----- Saving to disk ------------------------

def save_merges(merges, path):

    with open(path, 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(f'{a} {b}\n')

def save_vocab(string_to_index, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(string_to_index, f, ensure_ascii=False, indent=2)

#   ----- Main function -------------------------

def main(segmented_corpus, merges_output, vocab_output, threshold, vocab_size=20_000):

    lines = []
    with open(segmented_corpus, encoding='utf-8') as f:
        for sent in f:
            lines.extend(sent.split())

    print("Creating word objects")
    vocab = create_word_objects(lines)
    chars = {c for w in vocab for m in w.morphemes for c in m} 
    print("Learning bpe")
    merges = learn_bpe(vocab, threshold, vocab_size)
    save_merges(merges, merges_output)

    print("Building vocabulary")
    string_to_index, _ = build_vocab_from_merges(merges, chars)
    save_vocab(string_to_index, vocab_output)

if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]) if len(sys.argv) > 5 else 20_000,
    )






