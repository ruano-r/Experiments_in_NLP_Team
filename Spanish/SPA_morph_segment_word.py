import sys
from morphagram import MorphModel

#usage (bash)
#python SPA_morph_segment_word.py ~/Downloads/spa_1m.txt ~/Downloads/spa_1m_analyzed.txt

input_tok = sys.argv[1]
output_file = sys.argv[2]

SENTENCE_BREAK = False   # T = blank line between sentences, F = nothing

segmentation_model = MorphModel('morphagram_model.json')

cache = {}              #each unique word is segmented only once (much faster)

def analyze(word):      #takes one word instead of a sentence (compared to og)
    if word not in cache:
        cache[word] = segmentation_model.segment_text(word, '@@')
    return cache[word]

with open(input_tok, encoding="utf-8-sig") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        sentence = line.strip()
        if not sentence:
            continue                                  #skip empty lines (no stray blank lines)

        for word in sentence.split():               
            fout.write(analyze(word) + "\n")          #one word per line

        if SENTENCE_BREAK:          
            fout.write("\n")          # if setting on blank line marks the end of a sentence

print(f"Output file: {output_file}")
