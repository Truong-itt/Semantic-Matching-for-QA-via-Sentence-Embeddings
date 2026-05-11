"""
error_analysis.py
-----------------
Qualitative and quantitative analysis of sentence-selection errors.

Categories analysed
-------------------
1. Semantic paraphrase errors  – question uses different words from gold sentence
2. Lexical overlap bias        – wrong sentence shares many keywords with question
3. Long-context confusion      – misses that occur in longer paragraphs
4. Short-sentence bias         – gold sentence is unusually short
"""

from __future__ import annotations

import os
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"\b\w+\b", text.lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Error collection
# ---------------------------------------------------------------------------

def collect_errors(
    selector,
    samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run *selector* on *samples* and return the mis-classified ones.

    Each error record contains:
    - original sample fields
    - ``predicted`` : int   (predicted sentence index)
    - ``predicted_text`` : str
    - ``gold_text`` : str
    """
    errors = []
    for sample in samples:
        q     = sample["question"]
        sents = sample["sentences"]
        label = sample["label"]

        pred = selector.predict(q, sents)
        if pred != label and label < len(sents):
            errors.append({
                **sample,
                "predicted"     : pred,
                "predicted_text": sents[pred] if pred < len(sents) else "",
                "gold_text"     : sents[label],
            })
    return errors


# ---------------------------------------------------------------------------
# Category: Lexical overlap bias
# ---------------------------------------------------------------------------

def lexical_overlap_analysis(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check if the *predicted* (wrong) sentence has higher lexical overlap
    with the question than the *gold* sentence.

    Returns summary statistics.
    """
    pred_overlaps, gold_overlaps = [], []
    biased_count = 0

    for e in errors:
        q_toks    = _tokenize(e["question"])
        pred_toks = _tokenize(e["predicted_text"])
        gold_toks = _tokenize(e["gold_text"])

        pred_ov = _jaccard(q_toks, pred_toks)
        gold_ov = _jaccard(q_toks, gold_toks)

        pred_overlaps.append(pred_ov)
        gold_overlaps.append(gold_ov)

        if pred_ov > gold_ov:
            biased_count += 1

    return {
        "n_errors"               : len(errors),
        "lexical_bias_count"     : biased_count,
        "lexical_bias_rate"      : biased_count / max(len(errors), 1),
        "mean_pred_overlap"      : float(np.mean(pred_overlaps)) if pred_overlaps else 0.0,
        "mean_gold_overlap"      : float(np.mean(gold_overlaps)) if gold_overlaps else 0.0,
    }


# ---------------------------------------------------------------------------
# Category: Long context confusion
# ---------------------------------------------------------------------------

def context_length_analysis(
    errors: List[Dict[str, Any]],
    all_samples: List[Dict[str, Any]],
    bins: int = 5,
) -> Dict[str, Any]:
    """
    Compute per-bucket error rate by number of sentences in the context.
    """
    correct_lengths = [len(s["sentences"]) for s in all_samples
                       if s["id"] not in {e["id"] for e in errors}]
    error_lengths   = [len(e["sentences"]) for e in errors]

    all_lengths    = correct_lengths + error_lengths
    breakpoints    = np.percentile(all_lengths, np.linspace(0, 100, bins + 1))
    breakpoints    = np.unique(breakpoints)

    bucket_labels, bucket_err, bucket_tot = [], [], []
    for i in range(len(breakpoints) - 1):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        inc    = (lambda x: lo <= x <= hi) if i == len(breakpoints) - 2 else (lambda x: lo <= x < hi)
        tot    = sum(1 for n in all_lengths if inc(n))
        err    = sum(1 for n in error_lengths if inc(n))
        if tot > 0:
            bucket_labels.append(f"{int(lo)}-{int(hi)}")
            bucket_err.append(err)
            bucket_tot.append(tot)

    error_rates = [e / t for e, t in zip(bucket_err, bucket_tot)]
    return {
        "buckets"     : bucket_labels,
        "error_counts": bucket_err,
        "total_counts": bucket_tot,
        "error_rates" : error_rates,
    }


# ---------------------------------------------------------------------------
# Category: Semantic paraphrase difficulty
# ---------------------------------------------------------------------------

def paraphrase_difficulty(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic: a "paraphrase error" occurs when the gold sentence has low
    lexical overlap with the question (i.e. different wording).
    """
    low_overlap = [
        e for e in errors
        if _jaccard(_tokenize(e["question"]), _tokenize(e["gold_text"])) < 0.1
    ]
    return {
        "paraphrase_errors" : len(low_overlap),
        "paraphrase_rate"   : len(low_overlap) / max(len(errors), 1),
        "examples"          : low_overlap[:5],
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def run_error_analysis(
    selector,
    samples: List[Dict[str, Any]],
    save_dir: Optional[str] = None,
    tag: str = "model",
) -> Dict[str, Any]:
    """
    Run the complete error analysis suite.

    Parameters
    ----------
    selector  : selector object
    samples   : evaluation samples
    save_dir  : if given, save a JSON report and plots here
    tag       : label prefix for saved files

    Returns
    -------
    Full analysis dict
    """
    print(f"\n=== Error Analysis: {tag} ===")
    errors = collect_errors(selector, samples)
    n_total = len(samples)
    n_err   = len(errors)
    print(f"Total errors: {n_err}/{n_total}  ({100*n_err/n_total:.1f}%)")

    lex_stats = lexical_overlap_analysis(errors)
    ctx_stats = context_length_analysis(errors, samples)
    par_stats = paraphrase_difficulty(errors)

    print(f"  Lexical bias rate       : {lex_stats['lexical_bias_rate']:.3f}")
    print(f"  Paraphrase error rate   : {par_stats['paraphrase_rate']:.3f}")
    print(f"  Context-length buckets  : {list(zip(ctx_stats['buckets'], ctx_stats['error_rates']))}")

    report = {
        "tag"              : tag,
        "n_total"          : n_total,
        "n_errors"         : n_err,
        "error_rate"       : n_err / n_total,
        "lexical_overlap"  : lex_stats,
        "context_length"   : ctx_stats,
        "paraphrase"       : par_stats,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save JSON (exclude verbose examples)
        report_copy = {k: v for k, v in report.items() if k != "paraphrase"}
        report_copy["paraphrase"] = {k: v for k, v in par_stats.items() if k != "examples"}
        json_path = os.path.join(save_dir, f"{tag}_error_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_copy, f, indent=2, ensure_ascii=False)
        print(f"Report saved → {json_path}")

        # Plot: error rate by context length
        _plot_context_error_rate(ctx_stats, os.path.join(save_dir, f"{tag}_context_error.png"), tag)

    return report


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_context_error_rate(ctx_stats: Dict, save_path: str, tag: str):
    buckets     = ctx_stats["buckets"]
    error_rates = ctx_stats["error_rates"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(buckets, error_rates, color="salmon", edgecolor="darkred", alpha=0.85)
    ax.set_xlabel("Context length (# sentences)")
    ax.set_ylabel("Error rate")
    ax.set_title(f"Error rate by context length – {tag}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved → {save_path}")
