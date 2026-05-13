"""
supervised.py
-------------
Supervised sentence-selection models.

Approach: Binary classification on (Question, Sentence) pairs.

Feature vector for (q, sᵢ):
  - cosine similarity         : scalar  (1,)
  - element-wise product      : (D,)
  - absolute difference       : (D,)
  - concatenation [q; sᵢ]    : (2D,)
  → total feature dim = 1 + 3·D

Three classifiers are provided:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost  (requires ``xgboost`` package)

At inference, all sentences in a context are scored; the one with the
highest positive-class probability is selected.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Feature engineering

def build_pair_feature(
    q_vec: np.ndarray,
    s_vec: np.ndarray,
) -> np.ndarray:
    """
    Construct the feature vector for a (question, sentence) pair.

    Features
    --------
    cosine_sim  : (1,)
    elem_prod   : q · s  element-wise  (D,)
    abs_diff    : |q - s| (D,)
    concat      : [q, s]  (2D,)
    Total dimension: 1 + 3·D
    """
    cosine_sim = np.dot(q_vec, s_vec) / (
        np.linalg.norm(q_vec) * np.linalg.norm(s_vec) + 1e-10
    )
    elem_prod = q_vec * s_vec
    abs_diff  = np.abs(q_vec - s_vec)
    concat    = np.concatenate([q_vec, s_vec])

    return np.concatenate([[cosine_sim], elem_prod, abs_diff, concat])

def build_dataset(
    samples: List[Dict[str, Any]],
    encoder,
    neg_per_pos: int = 3,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) matrices for training a binary classifier.

    For each sample, the correct sentence → label 1.
    ``neg_per_pos`` randomly chosen wrong sentences → label 0.

    Parameters
    samples     : list of dicts with keys question / sentences / label
    encoder     : object with encode(List[str]) → np.ndarray
    neg_per_pos : negative examples per positive
    Returns
    X : (M, feature_dim)
    y : (M,)
    """
    rng  = np.random.default_rng(random_seed)
    X_rows, y_rows = [], []

    for sample in samples:
        q           = sample["question"]
        sents       = sample["sentences"]
        label       = sample["label"]
        n_sents     = len(sents)
        if label >= n_sents:
            continue
        all_texts = [q] + sents
        vecs      = encoder.encode(all_texts)
        q_vec     = vecs[0]
        s_vecs    = vecs[1:]
        # Positive pair
        X_rows.append(build_pair_feature(q_vec, s_vecs[label]))
        y_rows.append(1)
        # Negative pairs
        neg_indices = [i for i in range(n_sents) if i != label]
        if neg_indices:
            chosen = rng.choice(neg_indices, size=min(neg_per_pos, len(neg_indices)), replace=False)
            for ni in chosen:
                X_rows.append(build_pair_feature(q_vec, s_vecs[ni]))
                y_rows.append(0)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)

# Supervised Selector
class SupervisedSelector:
    """
    Sentence selector backed by a scikit-learn binary classifier.
    Parameters
    model_type : ``'lr'`` | ``'rf'`` | ``'xgb'``
    encoder    : encoder object (encode method)
    model_kwargs : extra kwargs passed to the underlying sklearn estimator
    """
    MODEL_REGISTRY = {
        "lr" : ("sklearn.linear_model", "LogisticRegression",
                dict(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)),
        "rf" : ("sklearn.ensemble",    "RandomForestClassifier",
                dict(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)),
        "xgb": ("xgboost",             "XGBClassifier",
                dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                     use_label_encoder=False, eval_metric="logloss",
                     random_state=42, n_jobs=-1)),
    }

    def __init__(self, model_type: str = "lr", encoder = None, **model_kwargs,):
        self.model_type  = model_type
        self.encoder     = encoder
        self._clf        = None
        self._model_kwargs = model_kwargs
        self._build_clf()

    def _build_clf(self):
        if self.model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"model_type must be one of {list(self.MODEL_REGISTRY)}.")
        module_name, cls_name, defaults = self.MODEL_REGISTRY[self.model_type]
        params = {**defaults, **self._model_kwargs}
        import importlib
        module = importlib.import_module(module_name)
        cls    = getattr(module, cls_name)
        self._clf = cls(**params)

    def train(self, samples: List[Dict[str, Any]], neg_per_pos: int = 3,):
        """
        Build feature matrix from *samples* and fit the classifier.
        """
        print(f"Building features for {len(samples)} samples …")
        X, y = build_dataset(samples, self.encoder, neg_per_pos=neg_per_pos)
        print(f"Feature matrix: {X.shape}  –  pos: {y.sum()}  neg: {(y==0).sum()}")
        self._clf.fit(X, y)
        print("Training complete.")
        return self

    def _score_sentences(
        self,
        question: str,
        sentences: List[str],
    ) -> np.ndarray:
        """Return P(positive) for each (question, sentence) pair."""
        all_texts = [question] + sentences
        vecs      = self.encoder.encode(all_texts)
        q_vec     = vecs[0]
        s_vecs    = vecs[1:]

        feats = np.stack([
            build_pair_feature(q_vec, s_vecs[i]) for i in range(len(sentences))
        ])
        proba = self._clf.predict_proba(feats)[:, 1]   # P(positive class)
        return proba

    def predict(self, question: str, sentences: List[str]) -> int:
        """Return the index of the selected sentence."""
        scores = self._score_sentences(question, sentences)
        return int(np.argmax(scores))

    def predict_topk(self, question: str, sentences: List[str], k: int = 3) -> List[int]:
        scores = self._score_sentences(question, sentences)
        k = min(k, len(sentences))
        return np.argsort(scores)[::-1][:k].tolist()

    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        top_k_values: List[int] = (1, 3, 5),
        verbose: bool = True,
    ) -> Dict[str, float]:
        n      = len(samples)
        hits   = {k: 0 for k in top_k_values}
        rr_sum = 0.0

        for sample in samples:
            q       = sample["question"]
            sents   = sample["sentences"]
            label   = sample["label"]
            scores  = self._score_sentences(q, sents)
            ranked  = np.argsort(scores)[::-1]
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
            tag  = self.model_type.upper()
            line = f"[{tag}]  " + "  ".join(
                f"Acc@{k}: {v:.4f}" for k, v in results.items() if k.startswith("Acc")
            )
            line += f"  MRR: {results['MRR']:.4f}"
            print(line)
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)
        print(f"Model saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            self._clf = pickle.load(f)
        print(f"Model loaded ← {path}")
        return self
