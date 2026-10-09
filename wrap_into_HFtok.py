# python wrap_into_HFtok.py VOCAB_JSON MERGES_TXT OUTPUT_DIR [--marker]
# --marker: only for tokenizers trained with a '_' word-start marker 
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, processors
from transformers import PreTrainedTokenizerFast
 
vocab_file, merges_file, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
use_marker = "--marker" in sys.argv[4:]
 
bpe = models.BPE.from_file(vocab_file, merges_file, unk_token="<unk>")
tok = Tokenizer(bpe)
 
# must match training (check that your segmented corpus was really lowercased)
tok.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
 
if use_marker:
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Metaspace(replacement="_", prepend_scheme="always", split=False),
    ])
else:
    tok.pre_tokenizer = pre_tokenizers.Whitespace()  # must match training
 
for t in ["<unk>", "<pad>", "<bos>", "<eos>"]:
    if tok.token_to_id(t) is None:
        raise ValueError(f"Special token {t} is not in the vocab file {vocab_file}")
 
tok.post_processor = processors.TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <eos> $B:1 <eos>:1",
    special_tokens=[("<bos>", tok.token_to_id("<bos>")), ("<eos>", tok.token_to_id("<eos>"))],
)
 
hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
)
hf_tok.save_pretrained(out_dir)