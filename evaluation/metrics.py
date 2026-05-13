"""
metrics.py
----------
Comprehensive evaluation metrics for sentence-selection QA.

Metrics implemented
-------------------
- Accuracy@k      (exact-match top-k)
- Recall@k        (any correct in top-k, alias for Acc@k in single-label setting)
- MRR             (Mean Reciprocal Rank)
- Precision / Recall / F1  (binary: predicted sentence vs gold sentence)
- Confusion matrix (per-sample rank distribution)
- Result table    (formatted comparison across multiple systems)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless rendering


# Core metric computation

def accuracy_at_k(ranked_lists: List[List[int]], labels: List[int], k: int) -> float:
    """
    Accuracy@k: fraction of samples where the gold label appears in top-k.

    Parameters
    ----------
    ranked_lists : list of ranked sentence indices per sample
    labels       : gold sentence index per sample
    k            : cut-off

    Returns
    -------
    float in [0, 1]
    """
    hits = sum(
        1 for ranked, label in zip(ranked_lists, labels)
        if label in ranked[:k]
    )
    return hits / len(labels)


def mean_reciprocal_rank(ranked_lists: List[List[int]], labels: List[int]) -> float:
    """Compute MRR across all samples."""
    rr_sum = 0.0
    for ranked, label in zip(ranked_lists, labels):
        if label in ranked:
            rank    = ranked.index(label) + 1
            rr_sum += 1.0 / rank
    return rr_sum / len(labels)


def precision_recall_f1(
    predictions: List[int],
    labels: List[int],
) -> Dict[str, float]:
    """
    Binary sentence-level P / R / F1.

    prediction == label   → True Positive
    prediction != label   → both a False Positive and a False Negative
    (each sample has exactly one correct sentence)
    """
    tp = sum(p == g for p, g in zip(predictions, labels))
    n  = len(labels)
    precision = tp / n
    recall    = tp / n
    f1        = tp / n   # all three collapse to accuracy in 1-of-N setting
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": tp / n}

def evaluate_selector(
    selector,
    samples: List[Dict[str, Any]],
    top_k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Full evaluation of a selector on *samples*.

    The *selector* must expose one of:
      - predict(question, sentences) → int
      - predict_topk(question, sentences, k) → List[int]

    Returns a flat dict of metric → value.
    """
    ranked_lists : List[List[int]] = []
    labels       : List[int]       = []
    predictions  : List[int]       = []

    max_k = max(top_k_values)
    for sample in samples:
        q     = sample["question"]
        sents = sample["sentences"]
        label = sample["label"]

        if hasattr(selector, "predict_topk"):
            ranked = selector.predict_topk(q, sents, k=max_k)
        else:
            pred   = selector.predict(q, sents)
            ranked = [pred]

        ranked_lists.append(ranked)
        labels.append(label)
        predictions.append(ranked[0])

    results: Dict[str, float] = {}
    for k in top_k_values:
        results[f"Accuracy@{k}"] = accuracy_at_k(ranked_lists, labels, k)
        results[f"Recall@{k}"]   = results[f"Accuracy@{k}"]  # identical in 1-of-N

    results["MRR"] = mean_reciprocal_rank(ranked_lists, labels)
    prf            = precision_recall_f1(predictions, labels)
    results.update(prf)
    return results

# Table formatting
def format_results_table(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
) -> str:
    """
    Format a comparison table.

    Parameters
    ----------
    results_dict : { system_name : { metric_name : value } }
    metrics      : subset of metrics to display (all if None)
    """
    if not results_dict:
        return ""

    # Collect all metric names
    all_metrics = metrics or sorted(
        set(m for d in results_dict.values() for m in d)
    )

    systems = list(results_dict.keys())
    col_w   = max(max(len(s) for s in systems), 20)
    m_w     = max(max(len(m) for m in all_metrics), 12)

    header = f"{'System':<{col_w}}" + "".join(f"  {m:>{m_w}}" for m in all_metrics)
    sep    = "-" * len(header)
    rows   = [header, sep]

    for system in systems:
        vals = results_dict[system]
        row  = f"{system:<{col_w}}"
        for m in all_metrics:
            v    = vals.get(m, float("nan"))
            row += f"  {v:>{m_w}.4f}"
        rows.append(row)

    return "\n".join(rows)

def save_results_csv(
    results_dict: Dict[str, Dict[str, float]],
    path: str,
):
    """Save the comparison table as a CSV file."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_metrics = sorted(set(m for d in results_dict.values() for m in d))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["System"] + all_metrics)
        for system, vals in results_dict.items():
            writer.writerow([system] + [f"{vals.get(m, float('nan')):.4f}" for m in all_metrics])
    print(f"Results CSV saved → {path}")

# Plotting
def plot_acc_at_k(
    results_dict: Dict[str, Dict[str, float]],
    k_values: Tuple[int, ...] = (1, 3, 5),
    save_path: Optional[str] = None,
    title: str = "Accuracy@k Comparison",
):
    """Bar-chart comparison of Accuracy@k across systems."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x       = np.arange(len(k_values))
    n_sys   = len(results_dict)
    width   = 0.8 / n_sys
    colors  = plt.cm.tab10.colors  # type: ignore

    for i, (system, vals) in enumerate(results_dict.items()):
        acc_vals = [vals.get(f"Accuracy@{k}", 0.0) for k in k_values]
        offset   = (i - n_sys / 2 + 0.5) * width
        bars     = ax.bar(x + offset, acc_vals, width, label=system, color=colors[i % 10])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()

def plot_mrr(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "MRR Comparison",
):
    """Horizontal bar chart of MRR."""
    systems = list(results_dict.keys())
    mrr_vals = [results_dict[s].get("MRR", 0.0) for s in systems]

    fig, ax = plt.subplots(figsize=(7, max(3, len(systems) * 0.6)))
    colors  = plt.cm.tab10.colors  # type: ignore
    y_pos   = np.arange(len(systems))

    bars = ax.barh(y_pos, mrr_vals, color=[colors[i % 10] for i in range(len(systems))])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(systems)
    ax.set_xlabel("MRR")
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, mrr_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()

def plot_confusion_rank_distribution(
    selector,
    samples: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Predicted Rank Distribution",
):
    """
    Histogram: at what rank does the gold sentence appear?
    """
    ranks = []
    for sample in samples:
        q     = sample["question"]
        sents = sample["sentences"]
        label = sample["label"]

        if hasattr(selector, "predict_topk"):
            ranked = selector.predict_topk(q, sents, k=len(sents))
        else:
            pred   = selector.predict(q, sents)
            ranked = [pred]

        if label in ranked:
            ranks.append(ranked.index(label) + 1)
        else:
            ranks.append(len(sents) + 1)
    max_rank = max(ranks)
    bins     = np.arange(0.5, min(max_rank + 1.5, 11.5), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ranks, bins=bins, edgecolor="black", color="steelblue", alpha=0.8)
    ax.set_xlabel("Rank of gold sentence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()
