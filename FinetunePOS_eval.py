### Evaluating model for POS - UD conllu format

# make sure you have all imports installed

# https://huggingface.co/datasets/universal-dependencies/universal_dependencies > look for your language

## python FinetunePOS_eval.py MODEL_FOLDER TOKENIZER_FOLDER OUTPUT_FOLDER LANGUAGE [opt. SEED]

## The script outputs a folder in which there are:
# - the finetuned model (and checkpoints)
# - tokenizer
# - the POS task classification report (sklearn style)

import sys
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed

if len(sys.argv) < 5:
    sys.exit("Call like: python FinetunePOS_eval.py MODEL_FOLDER TOKENIZER_FOLDER LANGUAGE [opt. SEED]")


MODEL_CHECKPOINT = sys.argv[1]
TOKENIZER_CHECKPOINT = sys.argv[2]
LANGUAGE = sys.argv[4] #UD name

#seed is optional so
if len(sys.argv) > 5:
    SEED = int(sys.argv[5])
    OUTPUT_DIR = sys.argv[3] + "_" + str(SEED)
else:
    SEED = 42 #or other random
    OUTPUT_DIR = sys.argv[3]
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

raw_datasets = load_dataset("universal-dependencies/universal_dependencies", LANGUAGE) #SPA "es_ancora"

#get labels from POS feature column (upos)
all_labels = set()
for split in raw_datasets:
    for seq in raw_datasets[split]["upos"]:
        all_labels.update(seq)

POS_LABELS = sorted(all_labels)
#print('Number and types of labels:')
#print(len(POS_LABELS), POS_LABELS)

#id mapping
label2id = {label: i for i, label in enumerate(POS_LABELS)}
id2label = {i: label for i, label in enumerate(POS_LABELS)}

print(f"Dataset loaded.")
#print(f"POS Column Name: '{"upos"}'")
#print(f"Train size: {len(raw_datasets['train'])}")
#print(f"Dev size: {len(raw_datasets['dev'])}")
#print(f"Test size: {len(raw_datasets['test'])}")

#get tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT) #add_prefix_space=True)

assert tokenizer.is_fast, "word_ids() requires fast tokenizer" #should be safe, wrapper is fast
assert tokenizer.pad_token is not None, "pad token missing" #should be safe, pad tokens should exist

#align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=min(512, model.config.max_position_embeddings),
    )

    labels = []
    for i, label_seq in enumerate(examples["upos"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)      #special tokens ignored in loss
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_seq[word_idx]])     #first subwords gets label
            else:
                label_ids.append(-100)      #other subwords ignored in loss
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

##load the model
#set_seed(SEED)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(POS_LABELS),
    id2label=id2label,
    label2id=label2id,
)
##after loading model you get some warnings: do not worry :).
#those are coming from the added layer that comes with AutoModelForTokenClassification

#make sure no mismatch embeddings (when combining different models and toks)
assert len(tokenizer) <= model.get_input_embeddings().num_embeddings, (
    f"Tokenizer has {len(tokenizer)} tokens but the model only has "
    f"{model.get_input_embeddings().num_embeddings} embeddings"
)
#mapping labels to all splits
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

#function to standard classification report
def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=-1)

    #???
    mask = labels != -100
    y_pred = preds[mask]
    y_true = labels[mask]

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f1,
    }


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

#parameters were taken from code from previous classes, maybe something is missing or too much!
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4, #TO DECIDE!
    weight_decay=0.01,
    load_best_model_at_end=True, #changeable!
    metric_for_best_model="f1_weighted",
    #seed=SEED, ##if we decide to run multiple seeds!
    report_to="none",  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

#the model will train and be evaluated on dev directly to monitor loss
trainer.train()
print('Model trained!')

# save best model (changeable!) and tokenizer configuration
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print('Model saved!')

test_out = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(test_out.predictions, axis=-1)
labels = test_out.label_ids
mask = labels != -100

report = classification_report(
    labels[mask], preds[mask],
    labels=list(range(len(POS_LABELS))),
    target_names=POS_LABELS,
    zero_division=0,
    digits=4,
)

#print(test_out.metrics)
#print(report)

#save report to outfile
with open(os.path.join(OUTPUT_DIR, "pos_test_report.txt"), "w") as f:
    f.write("Overall metrics:\n")
    f.write(json.dumps(test_out.metrics, indent=2))
    f.write("\n\nClassification report:\n")
    f.write(report)
