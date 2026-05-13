"""
convert_dataset.py
------------------
Converts the SQuAD v1.1 span-extraction dataset into a sentence-selection
format and persists the result to disk as JSON.

Each output sample has the shape:
{
    "id"       : str,
    "question" : str,
    "sentences": [str, str, ...],
    "label"    : int   # index of the sentence that contains the answer
}

Samples for which the answer sentence cannot be located are discarded.
"""

import json
import os
from typing import Any, Dict, List
from preprocessing.sentence_tokenizer import tokenize_sentences, find_answer_sentence_idx
from datasets import load_dataset

# Paths
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")

# Core conversion helpers
def convert_squad_sample(
    sample: Dict[str, Any],
    method: str = "nltk",
) -> Dict[str, Any] | None:
    """
    Convert a single SQuAD sample to sentence-selection format.

    Returns None if the answer sentence cannot be located.
    """
    context     = sample["context"]
    question    = sample["question"]
    answers     = sample["answers"]
    sample_id   = sample["id"]

    # Pick the first answer text
    if not answers["text"]:
        return None
    answer_text = answers["text"][0]

    sentences = tokenize_sentences(context, method=method)
    if not sentences:
        return None

    label = find_answer_sentence_idx(sentences, answer_text)
    if label == -1:
        return None

    return {
        "id"       : sample_id,
        "question" : question,
        "sentences": sentences,
        "label"    : label,
    }

def convert_split(
    split_data: List[Dict[str, Any]],
    method: str = "nltk",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert an entire dataset split.  Discards un-locatable samples.
    """
    converted, skipped = [], 0
    total = len(split_data)

    for i, sample in enumerate(split_data):
        result = convert_squad_sample(sample, method=method)
        if result is None:
            skipped += 1
        else:
            converted.append(result)
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{total} …")
    if verbose:
        print(f"  Done: {len(converted)} kept, {skipped} skipped.")
    return converted

# Main entry point
def load_and_convert(
    method: str = "nltk",
    max_train: int | None = None,
    max_val: int | None = None,
):
    """
    Load SQuAD from HuggingFace *datasets*, convert both splits, and save
    Parameters
    method    : str  – Sentence tokenisation method (``'nltk'`` | ``'regex'``).
    max_train : int  – Truncate training set (useful for quick tests).
    max_val   : int  – Truncate validation set.
    Returns
    train_data, val_data : List[dict], List[dict]
    """

    print("SQuAD v1.1 …")
    dataset = load_dataset("squad")
    train_raw = list(dataset["train"])
    val_raw   = list(dataset["validation"])
    if max_train:
        train_raw = train_raw[:max_train]
    if max_val:
        val_raw = val_raw[:max_val]

    print(f"\nConverting train split ({len(train_raw)} samples) …")
    train_data = convert_split(train_raw, method=method)
    print(f"\nConverting validation split ({len(val_raw)} samples) …")
    val_data = convert_split(val_raw, method=method)

    os.makedirs(PROC_DIR, exist_ok=True)
    train_path = os.path.join(PROC_DIR, "train.json")
    val_path   = os.path.join(PROC_DIR, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"\nTrain data saved → {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"Validation data saved → {val_path}")
    return train_data, val_data

def load_processed(split: str = "train") -> List[Dict[str, Any]]:
    """
    Load already-converted data from disk.
    Parameters
    split : ``'train'`` | ``'val'``
    """
    path = os.path.join(PROC_DIR, f"{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run convert_dataset.load_and_convert() first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    load_and_convert(save=True, max_train=1000, max_val=200)
