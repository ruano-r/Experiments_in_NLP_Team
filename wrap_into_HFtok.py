#   python wrap_into_HFtok.py VOCAB_JSON MERGES_TXT OUTPUT_DIR

import sys
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, processors
from transformers import PreTrainedTokenizerFast

vocab_file, merges_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

bpe = models.BPE.from_file(vocab_file, merges_file, unk_token="<unk>")
tok = Tokenizer(bpe)

#normalizer must match what was done at training time
tok.normalizer = normalizers.Sequence([
    normalizers.NFC(),
    normalizers.Lowercase(),
])

# Pre-tokenization has to match the training
tok.pre_tokenizer = pre_tokenizers.Whitespace()

# Special tokens must exist in the vocab, otherwise fail early with a clear message
for t in ["<unk>", "<pad>", "<bos>", "<eos>"]:
    if tok.token_to_id(t) is None:
        raise ValueError(f"Special token {t} is not in the vocab file {vocab_file}")

tok.post_processor = processors.TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <eos> $B:1 <eos>:1",
    special_tokens=[
        ("<bos>", tok.token_to_id("<bos>")),
        ("<eos>", tok.token_to_id("<eos>")),
    ],
)

#here we will add other special tokens
hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
)

hf_tok.save_pretrained(out_dir)
