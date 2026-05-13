"""
unsupervised.py
---------------
Unsupervised sentence-selection models based on vector similarity.

Two distance measures are implemented:
  - Cosine similarity  (higher → more similar)
  - Euclidean distance (lower  → more similar)

The embeddings can come from any encoder that exposes an
``encode(texts) → np.ndarray`` interface.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any

# Similarity functions
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vector *a* and each row of matrix *b*.

    Parameters
    a : (D,)       query vector
    b : (N, D)     sentence matrix
    Returns
    similarities : (N,)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm   # (N,)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Euclidean distances between vector *a* and each row of matrix *b*.
    Returns
    distances : (N,)  – smaller means more similar
    """
    return np.linalg.norm(b - a, axis=1)   # (N,)


# Unsupervised selector
class UnsupervisedSelector:
    """
    Select the best-matching sentence for a question using vector similarity.
    Parameters
    encoder  : object  –  Must expose ``encode(List[str]) -> np.ndarray``.
    metric   : str     –  ``'cosine'`` (default) | ``'euclidean'``.
    """

    def __init__(self, encoder, metric: str = "cosine"):
        self.encoder = encoder
        self.metric  = metric

    def score(
        self,
        question: str,
        sentences: List[str],
    ) -> np.ndarray:
        """
        Return a similarity / distance score for each sentence.
        For cosine  : higher score → better match.
        For euclidean : lower score → better match  (negated before return
                        so that higher is always better – consistent API).
        """
        texts    = [question] + sentences
        vecs     = self.encoder.encode(texts)
        q_vec    = vecs[0]            # (D,)
        s_vecs   = vecs[1:]           # (N, D)
        if self.metric == "cosine":
            return cosine_similarity(q_vec, s_vecs)
        elif self.metric == "euclidean":
            return -euclidean_distance(q_vec, s_vecs)   # negate so ↑ is better
        else:
            raise ValueError(f"Unknown metric: {self.metric!r}")

    def predict(
        self,
        question: str,
        sentences: List[str],
    ) -> int:
        """Return index of the best-matching sentence (highest score)."""
        scores = self.score(question, sentences)
        return int(np.argmax(scores))

    def predict_topk(
        self,
        question: str,
        sentences: List[str],
        k: int = 3,
    ) -> List[int]:
        """Return indices of the top-k best-matching sentences."""
        scores   = self.score(question, sentences)
        k        = min(k, len(sentences))
        top_k    = np.argsort(scores)[::-1][:k]
        return top_k.tolist()

    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        top_k_values: List[int] = (1, 3, 5),
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the selector on a list of samples.
        Each sample must have keys: ``question``, ``sentences``, ``label``.
        Returns a dict with Accuracy@k and MRR.
        """
        n = len(samples)
        hits   = {k: 0 for k in top_k_values}
        rr_sum = 0.0
        for sample in samples:
            q         = sample["question"]
            sents     = sample["sentences"]
            label     = sample["label"]
            scores    = self.score(q, sents)
            ranked    = np.argsort(scores)[::-1]
            for k in top_k_values:
                if label in ranked[:k]:
                    hits[k] += 1
            rank = int(np.where(ranked == label)[0][0]) + 1 if label < len(sents) else len(sents)
            rr_sum += 1.0 / rank
        results: Dict[str, float] = {}
        for k in top_k_values:
            results[f"Accuracy@{k}"] = hits[k] / n
        results["MRR"] = rr_sum / n
        if verbose:
            line = f"[{self.metric.upper()}]  " + "  ".join(
                f"Acc@{k}: {v:.4f}" for k, v in results.items() if k.startswith("Acc")
            )
            line += f"  MRR: {results['MRR']:.4f}"
            print(line)
        return results

# Comparison helper: Cosine vs Euclidean
def compare_metrics(
    encoder,
    samples: List[Dict[str, Any]],
    top_k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Dict[str, float]]:
    """
    Run both cosine and euclidean selectors and return a comparison dict.
    """
    results = {}
    for metric in ("cosine", "euclidean"):
        selector = UnsupervisedSelector(encoder, metric=metric)
        results[metric] = selector.evaluate(samples, list(top_k_values), verbose=True)
    return results
