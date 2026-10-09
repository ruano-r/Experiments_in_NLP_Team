# The bare minimum from MorphAGram's framework that is needed to use our segmentation model trained for Spanish
# Important code comes from https://github.com/rnd2110/MorphAGram/tree/0ccf074149baf78735c0f5adcc359a0f90e96f35

import operator
import json
from alphabet_detector import AlphabetDetector
from tokenizers.pre_tokenizers import Whitespace


def has_special_char(string: str) -> bool:
    """
    This function checks whether the given string is apt for our morphological segmentation function.
    :param string: the string to be segmented
    :return: True if the string only contains latin characters and no spaces
    """
    ad = AlphabetDetector()
    return any(c for c in string if not c.isalnum() and c.isspace()) and ad.is_latin(string)


def insert_splits(word, count, split_marker, solutions):
    # From the original MorphAGram as is
    """
    This function splits a given word into all the possible ways.
    The number of chunks is defined as "count".
    :param word: the word to split
    :param count: number of chunks
    :param split_marker: split marker
    :param solutions: output splits
    :return: output splits
    """

    # If count == 0, no more insertions necessary. Append current solution and return.
    if count == 0 and word not in solutions:
        solutions.append(word)
        return solutions
    if word in solutions:
        return solutions
    # Add a "+" in all possible places
    for i in range(len(word) + 1):
        # Construct new split.
        new_split = word[:i] + split_marker + word[i:]
        # Ignore instances of empty stems (for example: "e++xample" will be ignored).
        if split_marker + split_marker in new_split:
            continue
        # Call recursively with a decremented count.
        insert_splits(new_split, count - 1, split_marker, solutions)

    return solutions


class MorphModel:
    """
    Constructs a MorphAGram segmentation model from a json file, containing a list of:
        - word_segmentation_map
        - complex_nonterminal_counts
        - complex_nonterminal_compositions
        - prefix_suffix_compatibility
    which corresponds with the output of the function `parse_segmentation_output` from the original MorphAGram
    repository, plus an extra element that consists of a list of invariable words that won't be segmented by the model.
        - EXCLUDED_WORDS
    """

    def __init__(self, file):

        with open(file) as f:
            self.model = json.load(f)

    def segment_word(self, word):
        """
        This function takes a word and uses the MorphAGram model defining the class to morphologically segment the word.
        It is a slightly modified version of the segmentation function in MorphAGram's original work.

        How does the segmentation work?:
        If a word is seen in the MorphAGram model's training data, its segmentation is read from the segmentation output
        of the learning process. Otherwise, it's analyzed by finding the split that gives the highest MLE probability
        across its morphemes, along with the selection of compatible prefixes and suffixes. The information of the
        morphemes and their MLE probabilities and compatibility are driven from the segmentation output of the learning
        process.

        :param word: the word to be morphologically segmented
        :return: a list of all the morphemes in which the word was segmented
        """

        # the segmentation model is only trained for Spanish, so any word with special characters or in an alphabet
        # other than latin is returned as is
        if has_special_char(word):
            return word

        # read the segmentation model
        word_segmentation_map = self.model[0]
        complex_nonterminal_counts = self.model[1]
        complex_nonterminal_compositions = self.model[2]
        prefix_suffix_compatibility = self.model[3]
        EXCLUDED_WORDS = self.model[4]

        word_lower = word.lower()
        segmented_word = list()
        # If the word is too short, too long (more than 42 characters, iykyk),
        # or it belongs to the list of invariable words, do not segment it.
        if len(word) != len(word_lower) or word_lower in EXCLUDED_WORDS or len(word) < 3 or len(word) > 42:
            segmented_word.append(word)
        else:
            # Save the casing of all characters.
            casing = [ch != ch.lower() for ch in word]

            # If the word already exists in the segmentation map, replace with existing segmentation.
            analysis = {}
            if word_lower in word_segmentation_map:
                analysis = word_segmentation_map[word_lower]
            # If the word is not in the segmentation map, find the best segmentation for it.
            else:
                analysis['prefix'] = ''
                analysis['stem'] = word_lower
                analysis['suffix'] = ''
                max_score = 0
                segmentations = []
                # Generate al the possible splits
                insert_splits(word_lower, 2, '+', segmentations)
                for segmentation in segmentations:
                    segments = segmentation.split('+')
                    complex_prefix = segments[0]
                    complex_stem = segments[1]
                    complex_suffix = segments[2]
                    # Ignore words with unseen morphemes.
                    if (complex_prefix not in complex_nonterminal_counts['prefix'] or
                            complex_stem not in complex_nonterminal_counts['stem'] or
                            complex_suffix not in complex_nonterminal_counts['suffix']):
                        continue
                    # Ignore words with incompatible prefix and suffix.
                    if complex_suffix not in prefix_suffix_compatibility[complex_prefix]:
                        continue
                    # Calculate the probability.
                    # score=p(segmentation)=count(prefix)/count_of_all_prefixes+count(stem)/count_of_all_stems+count(suffix)/count_of_all_suffixes
                    complex_prefix_prop = complex_nonterminal_counts['prefix'][complex_prefix] / len(
                        word_segmentation_map)
                    complex_stem_prop = complex_nonterminal_counts['stem'][complex_stem] / len(
                        word_segmentation_map)
                    complex_suffix_prop = complex_nonterminal_counts['suffix'][complex_suffix] / len(
                        word_segmentation_map)
                    score = complex_prefix_prop * complex_stem_prop * complex_suffix_prop
                    # Keep track of the segmentation that gives the highest score.
                    if score > max_score:
                        analysis['prefix'] = max(complex_nonterminal_compositions['prefix'][complex_prefix].items(),
                                                 key=operator.itemgetter(1))[0] if len(complex_prefix) > 0 else ''
                        analysis['stem'] = max(complex_nonterminal_compositions['stem'][complex_stem].items(),
                                               key=operator.itemgetter(1))[0]
                        analysis['suffix'] = max(complex_nonterminal_compositions['suffix'][complex_suffix].items(),
                                                 key=operator.itemgetter(1))[0] if len(complex_suffix) > 0 else ''
                        max_score = score

            # Restore the actual casing.
            index = 0
            cased_analysis = {}
            for key in ['prefix', 'stem', 'suffix']:
                cased_morphs = []
                for morph in analysis[key].split():
                    cased_morph = ''
                    for ch in morph:
                        if casing[index]:
                            cased_morph += ch.upper()
                        else:
                            cased_morph += ch.lower()
                        index += 1
                    cased_morphs.append(cased_morph)
                    segmented_word.append(cased_morph)
                cased_analysis[key] = ' '.join(cased_morphs)

        return segmented_word

    def segment_text(self, text, segment_marker='+'):
        """
        This function takes a text and uses segment_word to segment it word by word using the MorphAGram model defining
        the class. It returns the segmented text, in which morpheme boundaries have been marked with segment_marker.

        :param text: text to be segmented
        :param segment_marker: the symbol to mark the separation between morphemes in a word
        :return: the segmented text
        """

        pre_tokenizer = Whitespace()
        pre_tokenized_text = pre_tokenizer.pre_tokenize_str(text)
        segmented_text = ''
        prev_offsets = (0,0)
        for word, offsets in pre_tokenized_text:
            segmented_word = segment_marker.join(self.segment_word(word))
            # add original whitespace before each segmented word
            segmented_text += text[prev_offsets[1]:offsets[0]] + segmented_word
            prev_offsets = offsets
        # in case there was some whitespace after last word
        segmented_text += text[offsets[1]:len(text)]

        return segmented_text
