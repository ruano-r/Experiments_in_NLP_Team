# Experiments_in_NLP_Team

The LLM maker script we use is an altered version of the one made by L.M. da Costa (LM-Maker). 

The data we use to train the tokenizers and the models in each language is from fine2web (for Finnish, Hungarian and Spanish) and fineweb (for English).



The pipeline for training/evaluating a tokenizer and model with our code is found below:

1- get data splits
use fineweb2_splits2.py to get the data splits, comments provide the REGEX for different languages:
    - 1M from train split for morphologically analysing and evaluating the tokenizer 
    - 100M from train split for training a small model
    - 1M from test split for other tests of the tokenizer

2- morphological analysis
run a morphogical analyser on the 1M train split in the language of choice. 
    For this project we used :
    - Finnish and English - uralicNLP morph analyser/segmenter https://github.com/mikahama/uralicNLP
    - Hungarian - emMorph https://github.com/nytud/emMorph
    - Spanish - morphtokenizer https://github.com/Albalbalba/morphtokenizer
    
3- process the output of the analyser to resturn one word per line, separated into morphemes by @@ (walked -- walk@@ed)
    - post processing scripts for the target languages are available within the folders of the same name in this repository.
    - for English, Finnish, and Hungarian, decisions had to be made regarding which possible segmentation was chosen per word:
        - the segmentation must be of the surface form (no canonical segmentation)
        - the option with most morphemes is chosen as an attempt to capture the most detail as possible, prefered over losing granularity, also given the posterior merges that can occur between those morphemes. it is easier to merge them than to split them at a late stage.

4- train tokenizer with training_tokenizer_v2.py
    - Decide on the BPE merge allowance threshold


5- convert the tokenizer into a huggingface tokenizer object for convenience and later use with wrap_into_HFtok.py

6- get statistics about the tokenizer with evals_tokenizer.py and complex_tokenizer_evals.py
    - evals_tokenizer.py - pass the 1M test split file, preprocessed, and the tokenizer
    - complex_tokenizer_evals.py - pass the 1M TRAIN data segmented with @@ as boundaries, and the tokenizer

7- train the llm with llm-maker_morph.py

8- evaluate model on dev/test with evals_model.py to get loss on dev/test + our main metrics like perplexity (not comparable across tokenisers) and bits-per-byte (comparable across tokenizers) + other (including next token accuracy)


9- evaluate the llm further with FinetunePOS_eval.py



    

    

