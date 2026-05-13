"""
run_unsupervised.py
End-to-end experiment runner for the unsupervised sentence-selection baseline.

Experiments performed
1. TF-IDF + Cosine similarity
2. TF-IDF + Euclidean distance
3. Sentence-BERT + Cosine similarity
4. Sentence-BERT + Euclidean distance
5. BM25 (sparse retrieval baseline)

All results are saved to results/tables/ and results/plots/.
"""

import os
import sys

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
from typing import Dict, List, Any

from preprocessing.convert_dataset import load_and_convert, load_processed
from embeddings.pretrained_encoder  import TFIDFEncoder, SBERTEncoder, BM25Encoder
from models.unsupervised             import UnsupervisedSelector, compare_metrics
from evaluation.metrics              import (
    evaluate_selector,
    format_results_table,
    save_results_csv,
    plot_acc_at_k,
    plot_mrr,
    plot_confusion_rank_distribution,
)
from evaluation.error_analysis import run_error_analysis

# Paths
RESULTS_DIR = os.path.join(ROOT, "results")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
K_VALUES    = (1, 3, 5)

# Data helpers
def get_data(max_train: int = 5000, max_val: int = 1000) -> tuple:
    """Load (or re-convert) the processed SQuAD data."""
    proc_train = os.path.join(ROOT, "data", "processed", "train.json")
    proc_val   = os.path.join(ROOT, "data", "processed", "val.json")

    if os.path.exists(proc_train) and os.path.exists(proc_val):
        print("Loading pre-processed data …")
        train = load_processed("train")[:max_train]
        val   = load_processed("val")[:max_val]
    else:
        train, val = load_and_convert(save=True, max_train=max_train, max_val=max_val)
    print(f"Train: {len(train)}  |  Val: {len(val)}")
    return train, val

# Build encoders
def build_tfidf_encoder(train_data: List[Dict], val_data: List[Dict]) -> TFIDFEncoder:
    """Fit TF-IDF encoder on all training texts."""
    all_texts = []
    for s in train_data + val_data:
        all_texts.append(s["question"])
        all_texts.extend(s["sentences"])
    enc = TFIDFEncoder(n_components=256)
    print("Fitting TF-IDF encoder …")
    enc.fit(all_texts)
    return enc

def build_sbert_encoder() -> SBERTEncoder:
    return SBERTEncoder(model_name="all-MiniLM-L6-v2")

# BM25 experiment (different API – per-sample evaluation)
def run_bm25(val_data: List[Dict], top_k_values=(1, 3, 5)) -> Dict[str, float]:
    from models.unsupervised import UnsupervisedSelector
    import numpy as np

    n      = len(val_data)
    hits   = {k: 0 for k in top_k_values}
    rr_sum = 0.0

    for sample in val_data:
        q     = sample["question"]
        sents = sample["sentences"]
        label = sample["label"]

        bm25_enc = BM25Encoder()
        bm25_enc.fit(sents)
        scores = bm25_enc.get_scores(q)
        ranked = np.argsort(scores)[::-1].tolist()

        for k in top_k_values:
            if label in ranked[:k]:
                hits[k] += 1

        rank = ranked.index(label) + 1 if label in ranked else len(sents) + 1
        rr_sum += 1.0 / rank

    results = {f"Accuracy@{k}": hits[k] / n for k in top_k_values}
    results.update({f"Recall@{k}": results[f"Accuracy@{k}"] for k in top_k_values})
    results["MRR"] = rr_sum / n
    results["precision"] = results["accuracy"] = results["f1"] = results["recall"] = results["Accuracy@1"]
    print(f"[BM25]  " + "  ".join(f"Acc@{k}: {results[f'Accuracy@{k}']:.4f}" for k in top_k_values)
          + f"  MRR: {results['MRR']:.4f}")
    return results

# Main experiment
def run_unsupervised_experiments(
    max_train: int = 5000,
    max_val:   int = 1000,
    run_error_analysis_flag: bool = True,
):
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    # Data
    train_data, val_data = get_data(max_train, max_val)
    # Encoders
    tfidf_enc = build_tfidf_encoder(train_data, val_data)
    # Experiments
    all_results: Dict[str, Dict] = {}

    for metric in ("cosine", "euclidean"):
        name     = f"TF-IDF + {metric.capitalize()}"
        selector = UnsupervisedSelector(tfidf_enc, metric=metric)
        print(f"\nEvaluating: {name}")
        res = evaluate_selector(selector, val_data, top_k_values=K_VALUES)
        print(f"  {res}")
        all_results[name] = res

    # BM25
    print("\nEvaluating: BM25")
    all_results["BM25"] = run_bm25(val_data, top_k_values=K_VALUES)
    # SBERT (optional – requires sentence-transformers)
    # if use_sbert:
    try:
        sbert_enc = build_sbert_encoder()
        for metric in ("cosine", "euclidean"):
            name     = f"SBERT + {metric.capitalize()}"
            selector = UnsupervisedSelector(sbert_enc, metric=metric)
            print(f"\nEvaluating: {name}")
            res = evaluate_selector(selector, val_data, top_k_values=K_VALUES)
            print(f"  {res}")
            all_results[name] = res
    except ImportError:
        print("sentence-transformers not installed – skipping SBERT experiments.")
    # Print comparison table
    print("\n" + "=" * 80)
    print("UNSUPERVISED EXPERIMENT RESULTS")
    print("=" * 80)
    display_metrics = [f"Accuracy@{k}" for k in K_VALUES] + ["MRR"]
    print(format_results_table(all_results, metrics=display_metrics))
    # Save results
    csv_path = os.path.join(TABLES_DIR, "unsupervised_results.csv")
    save_results_csv(all_results, csv_path)
    json_path = os.path.join(TABLES_DIR, "unsupervised_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results saved → {json_path}")
    # Plots
    plot_acc_at_k(
        all_results,
        k_values=K_VALUES,
        save_path=os.path.join(PLOTS_DIR, "unsupervised_acc_at_k.png"),
        title="Unsupervised Accuracy@k",
    )
    plot_mrr(
        all_results,
        save_path=os.path.join(PLOTS_DIR, "unsupervised_mrr.png"),
        title="Unsupervised MRR",
    )
    # Error analysis on TF-IDF cosine (best tfidf model)
    if run_error_analysis_flag:
        best_sel = UnsupervisedSelector(tfidf_enc, metric="cosine")
        run_error_analysis(
            best_sel,
            val_data,
            save_dir=os.path.join(RESULTS_DIR, "error_analysis"),
            tag="tfidf_cosine",
        )
    return all_results

if __name__ == "__main__":
    run_unsupervised_experiments(max_train=5000, max_val=500, use_sbert=False)
