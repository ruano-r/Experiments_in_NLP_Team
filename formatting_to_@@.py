import sys
from collections import defaultdict, Counter


INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]


def extract_pieces(analysis):
    """
    Extract possible morphs from words, removing the additional tags from EmMorph's structure

    Example:
        portás[/N]verseny[/N][Nom]
    -> ["portás", "verseny"]
    """
    #list for morphs, and string where the current morph is being built
    pieces = []
    current = ""

    #when an parenthesis is reached that morph is complete and can be added to the list 
    i = 0
    while i < len(analysis):

        if analysis[i] == "[":
            if current:
                pieces.append(current)
                current = ""

            end = analysis.find("]", i)
           
            if end == -1:
                break

            i = end + 1

        else:
            current += analysis[i]
            i += 1

    if current:
        pieces.append(current)

    return [piece for piece in pieces if piece]


def segmentation(analysis):
    """Builds the words back up from the morphs, joining them with @@"""

    pieces = extract_pieces(analysis)

    if not pieces:
        return None

    return "@@".join(pieces)


#store all possible analyses for each word
word_analyses = defaultdict(list)

#run functions on the data
with open(INPUT_FILE, "r", encoding="utf-8") as infile:

    for line in infile:

        line = line.rstrip("\n")

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        word = parts[0]
        analysis = parts[1]

        #ignore unknown words
        if "+?" in analysis:
            continue

        seg = segmentation(analysis)

        if seg:
            word_analyses[word].append(
                (seg, analysis)
            )


def choose_segmentation(word, analyses):
    """
    Choose the preferred segmentation per word 
    
    1. if there is only one segmentation, chose it
    2. if there are multiple, choose the most common option for segmentation for that word
    3. if there is a tie, choose the one with most morphemes (more fine grained one in theory)
    4. if that does not untie it, use the first one
    """

    segmentations = [seg for seg, analysis in analyses]

    counts = Counter(segmentations)

    # 1
    if len(counts) == 1:
        return segmentations[0]

    max_count = max(counts.values())
    # 2
    candidates = [
        seg
        for seg, count in counts.items()
        if count == max_count
    ]

    # 3
    max_parts = max(seg.count("@@") for seg in candidates)

    candidates = [
        seg
        for seg in candidates
        if seg.count("@@") == max_parts
    ]

    # 4
    for seg in segmentations:
        if seg in candidates:
            return seg


with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for word, analyses in word_analyses.items():

        selected = choose_segmentation(word, analyses)

        outfile.write(selected + "\n")