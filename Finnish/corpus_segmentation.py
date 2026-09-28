# SEGMENTING DATA FOR TRAINING TOKENIZER
#
# Fineweb2 Finnish data gets morphological segmented to create a training set for the morpho-aware tokenizer
# 
#



import re
import sys
import unicodedata

from collections import Counter
from datasets import load_dataset
from uralicNLP import uralicApi
from pyomorfi.omorfi import Omorfi
from pyomorfi.omorfi.token import Token


# ----- Config --------------------------------------

TARGET = 1_000_000
WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE) # only words, remove punctuation (keep digits, for years) --> right now only words consisting of just letters and tokens consisting of just digits are kept
FINNISH_WORD_RE = re.compile(r"^[a-zA-ZäöåÄÖÅ]+$")

omorfi = Omorfi()
omorfi.load_segmenter(r"C:\Users\Liann\Documents\Lianne\VU\2.1_Experiments_in_NLP\experiments_venv\Lib\site-packages\pyomorfi\omorfi.segment.kfst")

#download this once
#uralicApi.download("fin")

# ----- Prepare Data for Segmentation ---------------

def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return text.strip()

def tokenize_words(text):
    words = WORD_RE.findall(text)
    return [w for w in words if FINNISH_WORD_RE.match(w)] #only words containing characters that are allowed in Finnish are returned 

def quality_check(doc):

     language_score = doc.get("language_score", 1.0)
     if language_score < 0.95:
          return False
     return True

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

        text = normalize(doc["text"])
        words = tokenize_words(text)

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

            if word.isdigit():
                segment = [word]  # treat whole number as one morpheme
            elif analyser == "pyomorfi":
                segment = segment_pyomorfi(word)
            elif analyser == "uralicnlp":
                segment = segment_uralicnlp(word)
            else:
                raise ValueError(
                    f"Unknown analyser: {analyser}"
                )

            if segment is None:
                cache[word] = None
                continue

            cache[word] = parse_segment(segment)


        collected_segmentations.append(cache[word])

     return collected_segmentations

def parse_segment(segment):

     return "@@".join(segment)

# ----- Pyomorfi -------------------------------------------

def segment_pyomorfi(word):

    tok = Token(word.lower())
    analyses = omorfi.segment(tok)

    if not analyses:
        return None

    
    # first analysis is usually the most likely one?
    return analyses[0].get_segments()
     

# ----- UralicNLP ---------------------------------

def segment_uralicnlp(word):    #Takes longer than pyomorfi, results seem to be worse


    segmentations = uralicApi.segment(word.lower(), "fin")

    if len(segmentations) == 0:
         return [word]

    segments = min(segmentations,key=len)

    return segments

# ----- Save to Disk -------------------------------

def write_output(filepath, data, analyser):

     with open(filepath, 'w', encoding='utf-8') as f:
          f.write(f"#analyser used: {analyser}\n")
          for item in data:
               f.write(f"{item}\n")

# ----- Main ---------------------------------------

def main(analyser="pyomorfi"):

    #load dataset
    print("Collecting data sample")
    dataset = load_dataset("HuggingFaceFW/fineweb-2", name="fin_Latn", split="train", streaming=True)
    sample = collect_sample(dataset)
    types = unique_words(sample)
    print(f"Total tokens in sample {TARGET}, number of distinct tokens: {len(types)}")


    #segmenting
    print("Segmenting sample")
    segmentations = segment_sample(sample, analyser)


    output_file = "segmented_pretraining.txt"
    write_output(output_file, segmentations, analyser)
    print("Output has been written to disk") 



    # do something

if __name__ == "__main__":
   main(
        sys.argv[1]
   )