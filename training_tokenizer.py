#   MORPHO-AWARE BPE TOKENIZER
#
#   This tokenizer is an adaptation of the known BPE algorithm and has an additional variable that can be tuned for strictness and cross-boundary morpheme merging. 
#
#   Based on: https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/

import sys
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

    word_freq = Counter(lines)
    vocab = []

    for token, freq in word_freq.items():
        morphs = token.split('@@')                  #['walk', 'ed']
        morphemes = [list(m) for m in morphs]       #[['w', 'a', 'l', 'k'],['e', 'd']]
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
    Each occurrence is tagged on whether it is an intra-morpheme pair or an inter-morpheme (cross-boudnary) pair.
    """
    pair_counts = Counter()
    pair_kind = defaultdict(set)        # pair -> {'intra', 'inter'}

    for word in vocab:
        morphemes = word.morphemes

        for m_idx, morph in enumerate(morphemes):       # loop over morphemes in Word

            # intra-morpheme pairs        
            for i in range(len(morph) - 1):
                pair = (morph[i], morph[i+1])
                pair_counts[pair] += word.freq
                pair_kind[pair].add('intra')

            # inter-morpheme pairs: cross boundary pair
            # only possible if both morphemes are already 'complete'
            if m_idx + 1 < len(morphemes):      # check if there is a morpheme ofter the current one
                nxt = morphemes[m_idx + 1]
                if is_complete(morph) and is_complete(nxt):
                    pair = (morph[0], nxt[0])
                    pair_counts[pair] += word.freq
                    pair_kind[pair].add('inter')
            
    return pair_counts, pair_kind 


#   ----- Training BPE -------------------------------------------

def learn_bpe(vocab, threshold, vocab_size): #INCORPORATE THE THRESHOLD IN HERE
    """
    FUNCTION DESCRIPTION
    The apply_bpe() function applies the learned merge rules to a new word
    """

    merges = []

    for _ in range(vocab_size):
        pair_counts, pair_kind = get_pair_frequencies(vocab)

        if not pair_counts:
            break

        # filter all the cross-boundary pairs that don't meet the threshold

        allowed = {}
        for pair, count in pair_counts.items():
            kinds = pair_kind[pair]
            if kinds == {'inter'}:
                if count <= threshold:
                    continue                # frequency is lower than threshold --> skip
                allowed[pair] = count
            if kinds == {'intra'}:          # intra is always allowed
                allowed[pair] = count               

                # right now a pair could be both inter and intra, and thus in theory sneak through the threshold --> might need to change this
                # morphemes can keep on merging (a + b + c --> ab + c --> abc), i think that is fine, but this is a choice

        if not allowed:     #if no allowed pairs left, break
            break

        most_frequent = max(allowed, key=allowed.get)

        merges.append(most_frequent)

        apply_merge(vocab, most_frequent)

    return merges


#   ----- Merging pairs ---------------------------------------------

def apply_merge(vocab, pair):
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
    FUNCTION DESCRIPTION
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

def build_vocab_from_merges(merges, specials=('<unk>', '<pad>', '<bos>', '<eos>')):

    symbols = set()
    for a, b in merges:
        symbols.add(a)
        symbols.add(b)
        symbols.add(a + b)
    index_to_string = list(specials) + sorted(symbols)
    string_to_index = {tok: i for i, tok in enumerate(index_to_string)}

    return string_to_index, index_to_string

#   ----- Saving to disk ------------------------

def save_merges(merges, path):

    with open(path, 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(f'{a} {b}\n')

def save_vocab(string_to_index, path):
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(string_to_index, f, ensure_ascii=False, indent=2)


def load_merges(path):
    """
    Loading merges again --> not necessary in this script, but useful when you need to run the tokenizer again
    """

    merges = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            a, b = line.rstrip('\n').split(' ')
            merges.append((a, b))
    return merges

#   ----- Main function -------------------------

def main(segmented_corpus, merges_output, vocab_output, threshold, vocab_size=20_000):

    lines = []
    with open(segmented_corpus, encoding='utf-8') as f:
        for sent in f:
            lines.extend(sent.split())

    print("Creating word objects")
    vocab = create_word_objects(lines)
    print("Learning bpe")
    merges = learn_bpe(vocab, threshold, vocab_size)
    save_merges(merges, merges_output)

    print("Building vocabulary")
    string_to_index, _ = build_vocab_from_merges(merges)
    save_vocab(string_to_index, vocab_output)

if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]) if len(sys.argv) > 4 else 20_000,
    )






