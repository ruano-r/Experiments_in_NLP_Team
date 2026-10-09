# to get splits from fineweb2 
# with this script you can download any size of splits from the HF dataset,
# according to needs for train, dev and test
#call: python fineweb2_splits.py TARGET_WORDS OUT_PATH LANG_CODE DATASET_SPLIT REMOVE_PUNCT_DIGITS(1/0) (opt. SKIP_WORDS)

#FOR TRAIN TOK 1M python fineweb2_splits.py 1000000 tokenizer.txt spa_Latn train 1
#FOR TRAIN MODEL 100M python fineweb2_splits.py 100000000 model_train.txt spa_Latn train 0 
#FOR DEV python fineweb2_splits.py DEV_SIZE dev.txt spa_Latn train 0 100000000
#FOR TEST python fineweb2_splits.py TEST_SIZE test.txt spa_Latn test 0


# ---- imports
import re
import unicodedata
from datasets import load_dataset
import sys

if len(sys.argv) not in (6,7):
        print("usage: python fineweb2_splits.py TARGET_WORDS OUT_PATH LANG_CODE DATASET_SPLIT REMOVE_PUNCT(1/0)[opt. SKIP_WORDS]")
        sys.exit(1)
        
# ---- args
TARGET_WORDS = int(sys.argv[1])
OUT_PATH = sys.argv[2]
LANG_CODE = sys.argv[3]
DATASET_SPLIT = sys.argv[4] #train or test > to get dev use train
REMOVE_PUNCT_DIGITS = sys.argv[5] == "1" #1 = remove punct, symb & digits
SKIP_WORDS = int(sys.argv[6]) if len(sys.argv) == 7 else 0

# ---- editable settings
MIN_LANG_SCORE = 0.95
SKIP_MARGIN = 1_000_000 #!!!!make sure train and dev do not overlap


# ---- cleaning
EMAIL_RE = re.compile(r"[\w\.\-+%]+@[\w\-]+(?:\.[\w\-]+)+") # remove emails
# remove https, www etc
URL_RE = re.compile(r"(?:https?://|ftp://|www\.)\S+") 

########## EDIT alphabet ACCORDING TO LANGUAGE!

#anything not in this set is removed (emojis, other alphabets)
#i'm keeping alphabet, spanish accents, digits, whitespace, punct
NOTALLOW_RE = re.compile(
    r"[^a-z0-9áéíóúüñ\s"
    r"\.,;:!?¿¡\"'()\[\]{}\-–—…/\\%&@#+*=<>_$€«»“”‘’°ºª|~^`]")

SPACES_RE = re.compile(r"[ \t\u00a0]+")
# i will want to have sentences per line
SENT_END_RE = re.compile(r"([.!?]+[\"'»”’)\]]*) +")

#punctuation symbols and digits
PUNCT_DIGITS_RE = re.compile(r"[^\w\s]|[\d_]")

#ONLY words are added to the count, no numbers no punctuation
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE) 

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text) 
    text = text.lower()
    text = EMAIL_RE.sub(" ", text) 
    text = URL_RE.sub(" ", text)
    text = SPACES_RE.sub(" ", text)
    text = SENT_END_RE.sub(r"\1\n", text)      #new line for each sentence (. ! ? followed by whitespace) > will split mr. /n big
    text = SPACES_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not NOTALLOW_RE.search(ln)]
    text = "\n".join(lines)
    if REMOVE_PUNCT_DIGITS:
        text = PUNCT_DIGITS_RE.sub(" ", text) #punct symb and digits are space
        text = SPACES_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)  # drop empty lines

# ---- stream dataset
ds = load_dataset(
    "HuggingFaceFW/fineweb-2",
    name=LANG_CODE,
    split=DATASET_SPLIT,
    streaming=True,
)

# ---- skipping if needed
skip_target = SKIP_WORDS + (SKIP_MARGIN if SKIP_WORDS else 0)
skipping = skip_target > 0
skipped_words = 0
total_words = 0
docs = 0
seen = 0
low_score = 0

# ---- save to outfile
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for row in ds:
        seen += 1
        if row["language_score"] <= MIN_LANG_SCORE:
            low_score += 1
            continue

        cleaned = clean_text(row["text"])
        if not cleaned:
            continue
        n = len(WORD_RE.findall(cleaned))
        if n == 0:
            continue

        if skipping:
            skipped_words += n
            if skipped_words >= skip_target:
                skipping = False
            continue

        f.write(cleaned + "\n")        # one sent per line
        total_words += n
        docs += 1
        if total_words >= TARGET_WORDS:
            break

print(f"{docs:,} docs, {total_words:,} words")
print("Saved to:", OUT_PATH)
