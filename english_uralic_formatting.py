##changing lianne's script in order to fit the english morph analyser's structure

# SEGMENTING ENGLISH DATA FOR TRAINING THE MORPHO-AWARE TOKENIZER
#
# Usage:
#   python segment_english.py uralicnlp OUTPUT_FILE
#
# Takes ~1M words from FineWeb, segments each word with uralicNLP's English
# analyser and writes one lowercased, @@-joined segmentation per line.

import re
import sys
import unicodedata

from collections import Counter
from datasets import load_dataset
from uralicNLP import uralicApi


# ----- Config --------------------------------------

TARGET = 1_000_000

# CHECK THIS: FineWeb-2 may not contain English. If load_dataset fails, use
# DATASET = "HuggingFaceFW/fineweb", CONFIG = "sample-10BT"
DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "eng_Latn"

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)   # letter-only words, punctuation and digits dropped
ENGLISH_WORD_RE = re.compile(r"^[a-zA-Z]+$")     # only plain English letters


# ----- Prepare Data for Segmentation ---------------

def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def tokenize_words(text):
    words = WORD_RE.findall(text)
    return [w.lower() for w in words if ENGLISH_WORD_RE.match(w)]


def quality_check(doc):
    language_score = doc.get("language_score", 1.0)
    return language_score >= 0.95


def collect_sample(dataset):
    seen = set()
    total = 0
    collected_tokens = []

    for doc in dataset:
        if total >= TARGET:
            break

        if not quality_check(doc):
            continue

        # simple (not complete) way to check duplicates
        doc_hash = hash(doc["text"][:500])
        if doc_hash in seen:
            continue
        seen.add(doc_hash)

        words = tokenize_words(normalize(doc["text"]))

        collected_tokens.extend(words)
        total += len(words)

    return collected_tokens[:TARGET]


# ----- Segmentation ----------------------------------------

def unique_words(sample):
    return Counter(sample)


def parse_segment(segment):
    return "@@".join(segment)


def segment_sample(sample, analyser):
    collected_segmentations = []
    cache = {}

    for word in sample:

        if word not in cache:
            if analyser == "uralicnlp":
                segment = segment_uralicnlp(word)
            else:
                raise ValueError(f"Unknown analyser: {analyser}")

            cache[word] = None if segment is None else parse_segment(segment)

        # unknown words are skipped every time they occur
        if cache[word] is not None:
            collected_segmentations.append(cache[word])

    return collected_segmentations


# ----- UralicNLP (English) ---------------------------------

def parse_analysis(analysis):
    """splits the morphological analysis token structure 
    'UN+happy[ADJ]+NESS[ADJ/N]+N' -> ['un', 'happy', 'ness']"""
    
    morphs = []
    root_seen = False
    for tok in analysis.split("+"):
        name = tok.split("[")[0]
        if not name:
            continue
        if not root_seen:
            # prefix or root
            morphs.append(name.lower())          
            if name[0].islower():
                root_seen = True
        # derivational suffix
        elif "[" in tok:
            morphs.append(name.lower())          
        #  inflection/POS tag will be ignore
    return morphs





def align(word, morphs):
    """making the segments into lists"""
    if not word.startswith(morphs[0]):
        return None
    pieces, pos = [], 0
    for m in morphs[1:]:
        cut = word.find(m, pos + 1)
        if cut == -1:
            return None                     
        # morph is respelled (happy/happi, go/went)
        pieces.append(word[pos:cut])
        pos = cut
    chunk, last = word[pos:], morphs[-1]
    # leftover letters = inflection (cat|s, walk|ed)
    if len(chunk) > len(last):                
        pieces += [last, chunk[len(last):]]
    else:
        pieces.append(chunk)
    return pieces


def segment_uralicnlp(word):
    """choosing the analysis that keeps the surface form, and is the most fine grained"""
    word = word.lower()
    analyses = uralicApi.analyze(word, "eng")
    if not analyses:
        return None                           
    # unknown word skip

    options = []
    for a, _ in analyses:
        morphs = parse_analysis(a)
        if morphs:
            pieces = align(word, morphs)
            if pieces and "".join(pieces) == word:
                options.append(pieces)

    if not options:
        return [word]                         
    # if no surface-faithful option, keep the whole word
    # finest grained among the faithful ones
    return max(options, key=len) 



# ----- Save to Disk -------------------------------

def write_output(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(f"{item}\n")


# ----- Main ---------------------------------------

def main(analyser, output_file):
    print("Collecting data sample")
    dataset = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    sample = collect_sample(dataset)
    types = unique_words(sample)
    print(f"Total tokens in sample {len(sample)}, number of distinct tokens: {len(types)}")

    print("Segmenting sample")
    segmentations = segment_sample(sample, analyser)

    write_output(output_file, segmentations)
    print(f"Output has been written to {output_file}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])