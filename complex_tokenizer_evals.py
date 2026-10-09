"""python eval_morph_boundaries.py TOKENIZER_DIR SEGMENTED_FILE [--name RUN_NAME]
 SEGMENTED_FILE: the @@-segmented tokenizer training data (walk@@ed). 
 

What is this test trying to answer? how many of the morpheme boundaries in our training data
does each tokenizer keep?


1. Read the @@-segmented training data (e.g. walk@@ed)
2. For each word, remove the @@ to get the clean word ("walked") and tokenize it with the wrapped tokenizer
3. Gold boundaries: the positions where one morpheme ends and the next starts in the original segmentation used for training
4. Predicted boundaries is the index of where the boundary is in the word (_wa|lk|ed -> positions 2 and 4).
5. a gold boundary has been preserved if the tokenizer also cuts there, and changed if a token crosses it (e.g. wa|lke|d has no cut at 4).

Output is reported either for types and for tokens
- boundary preservation rate- how many gold boundaries were kept or lost in percentage
- morphemes intact- amount of morphemes that are the same in the gold and newly tokenized data
- number of morphemes that are exactly one token (cant cross a morpheme boundary)
- words fully preserved, and amount of words losing 0, 1, 2, 3+ boundaries
- extra boundaries per word- intermorpheme cuts

."""
 
import os, json, argparse
from collections import Counter
from transformers import AutoTokenizer


def load_words(path):
    counter = Counter()
    with open(path, encoding="utf-8") as file:
        for line in file:
            counter.update(line.split())
    return counter


def evaluate(word_freq, tokenizer):
     #getting the parts of words, needed for later comparison and for recreating the full word
    words = [w for w in word_freq if any(w.split("@@"))]
    #add all morph lists to larger list of all words, per word
    morph_lists = [[m for m in w.split("@@") if m] for w in words]
    # type = every unique word counts once, token = weighted by word frequency
    acc = {"type": Counter(), "token": Counter()}
    for w, morphs in zip(words, morph_lists):
        #word frequency
        freq = word_freq[w]
        #rebuild the clean word and get its total length in characters
        clean = "".join(morphs)
        L = len(clean)
        #gold boundaries: positions where one morpheme ends and the next starts
        gold, pos = set(), 0
        for m in morphs[:-1]:
            pos += len(m)
            gold.add(pos)
        #tokenize the clean word with the wrapped tokenizer
        toks = tokenizer.tokenize(clean)
        #dropping the "_" marker for comparison, so token lengths match the real text
        if toks:
            toks[0] = toks[0].lstrip("_")
        #predicted boundaries: positions wherea  token ends and the next starts
        pos, pred, spans = 0, set(), set()
        for t in toks:
            # needed for "morpheme is exactly one token"
            spans.add((pos, pos + len(t)))   
            pos += len(t)
            if 0 < pos < L:                 
                # skip the word end "boundary" because we only care about the ones in the middle of the word
                pred.add(pos)
        lost, extra = gold - pred, pred - gold

        edges = [0] + sorted(gold) + [L]
        n_intact = n_single = 0
        for s, e in zip(edges, edges[1:]):
            if (s == 0 or s in pred) and (e == L or e in pred):
                n_intact += 1
                n_single += (s, e) in spans

        for key, wt in (("type", 1), ("token", freq)):
            #counting dictionaries per statistic
            d = acc[key]
            d["words"] += wt
            d["extra_boundaries"] += wt * len(extra)
            if len(morphs) > 1:
                d["multi_words"] += wt
                d["gold"] += wt * len(gold)
                d["lost"] += wt * len(lost)
                d["morphemes"] += wt * len(morphs)
                d["morphemes_intact"] += wt * n_intact
                d["morphemes_single_token"] += wt * n_single
                d["words_fully_preserved"] += wt * (not lost)
                d["sum_loss_fraction"] += wt * len(lost) / len(gold)
                d[f"lost_{min(len(lost), 3)}"] += wt


    #writting the direct stat counts to out dictionary and calculating the rest during while appending te entry
    out = {}
    for key, d in acc.items():
        g, mw, mo = d["gold"], d["multi_words"], d["morphemes"]
        out[key] = {
            "words": d["words"],
            "multi_morpheme_words": mw,
            "gold_boundaries": g,
            "boundary_preservation_rate": 1 - d["lost"] / g if g else None,
            "boundary_change_rate": d["lost"] / g if g else None,
            "morphemes_intact_rate": d["morphemes_intact"] / mo if mo else None,
            "morphemes_single_token_rate": d["morphemes_single_token"] / mo if mo else None,
            "words_fully_preserved_rate": d["words_fully_preserved"] / mw if mw else None,
            "mean_word_loss_fraction": d["sum_loss_fraction"] / mw if mw else None,
            "words_losing_0_1_2_3plus_boundaries": [d[f"lost_{i}"] / mw if mw else None for i in range(4)],
            "extra_boundaries_per_word": d["extra_boundaries"] / d["words"],
        }
    return out

# printing the stats

def print_results(r):
    for key, title in (("type", "UNIQUE WORDS"), ("token", "ALL WORD OCCURRENCES (frequency-weighted)")):
        x = r[key]
        print("\n" + "=" * 60 + f"\nMORPH BOUNDARIES - {title}\n" + "=" * 60)
        print(f"Words: {x['words']:,} | multi-morpheme: {x['multi_morpheme_words']:,} | "
              f"gold boundaries: {x['gold_boundaries']:,}")
        print(f"Boundaries preserved: {x['boundary_preservation_rate']*100:.2f}%   "
              f"changed: {x['boundary_change_rate']*100:.2f}%")
        print(f"Morphemes intact (edges on token boundaries): {x['morphemes_intact_rate']*100:.2f}%")
        print(f"Morphemes that are exactly one token:         {x['morphemes_single_token_rate']*100:.2f}%")
        print(f"Words fully preserved: {x['words_fully_preserved_rate']*100:.2f}% | "
              f"mean per-word loss: {x['mean_word_loss_fraction']*100:.2f}%")
        h = x["words_losing_0_1_2_3plus_boundaries"]
        print("Words losing 0/1/2/3+ boundaries: " + " / ".join(f"{v*100:.1f}%" for v in h))
        print(f"Extra boundaries (cuts inside morphemes) per word: {x['extra_boundaries_per_word']:.3f}")


#better option than sys to keep account of what argument is what
def main():
    p = argparse.ArgumentParser()
    p.add_argument("tokenizer_dir")
    p.add_argument("segmented_file")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    results = evaluate(load_words(a.segmented_file), tokenizer)
    print_results(results)

    name = a.name or os.path.basename(os.path.normpath(a.tokenizer_dir))
    out = f"morph_eval_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()