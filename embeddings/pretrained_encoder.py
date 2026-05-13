"""
pretrained_encoder.py
---------------------
Sentence encoder backed by pre-trained transformer checkpoints
(Sentence-BERT family via ``sentence-transformers``) with a
TF-IDF/SVD fallback for resource-constrained environments.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

# Sentence-BERT Encoder
class SBERTEncoder:
    """
    Thin wrapper around a ``sentence-transformers`` model.

    Parameters
    ----------
    model_name : HuggingFace model name / local path.
    batch_size : Inference batch size.
    device     : ``'cpu'`` | ``'cuda'`` | ``'auto'``.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = "auto",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers:  pip install sentence-transformers"
            ) from exc

        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model      = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.model_name = model_name

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode *texts* into L2-normalised dense vectors.

        Returns
        -------
        np.ndarray  shape (N, D)
        """
        embeddings = self.model.encode(
            texts,
            batch_size       = self.batch_size,
            show_progress_bar= show_progress,
            normalize_embeddings=True,
            convert_to_numpy = True,
        )
        return embeddings  # (N, D)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

# TF-IDF + Truncated SVD Encoder (no GPU / no heavy dependencies)
class TFIDFEncoder:
    """
    Lightweight encoder: TF-IDF vectoriser → Truncated SVD projection.
    Useful as a fast baseline or when GPU memory is limited.

    Parameters
    n_components : Latent dimensions after SVD (output vector size).
    max_features : TF-IDF vocabulary cap.
    """
    def __init__(
        self,
        n_components: int = 256,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer

        self.n_components = n_components
        self._fitted      = False
        self._pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
            )),
            ("svd",   TruncatedSVD(n_components=n_components, random_state=42)),
            ("norm",  Normalizer(copy=False)),
        ])

    def fit(self, texts: List[str]) -> "TFIDFEncoder":
        """Fit the TF-IDF + SVD pipeline on *texts*."""
        self._pipe.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        """Transform *texts* → vectors of shape (N, n_components)."""
        if not self._fitted:
            raise RuntimeError("TFIDFEncoder must be fit() before encode().")
        return self._pipe.transform(texts).astype(np.float32)

    def fit_encode(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one call."""
        return self._pipe.fit_transform(texts).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# BM25-based Encoder (sparse retrieval baseline)
class BM25Encoder:
    """
    BM25 scoring as a retrieval baseline (non-neural).
    Requires ``rank_bm25``:  pip install rank-bm25
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as e:
            raise ImportError("pip install rank-bm25") from e

        self._BM25Okapi = BM25Okapi
        self.k1 = k1
        self.b  = b
        self._bm25: Optional[object] = None
        self._corpus_tokens: Optional[List[List[str]]] = None

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, sentences: List[str]) -> "BM25Encoder":
        self._corpus_tokens = [self._tokenize(s) for s in sentences]
        self._bm25 = self._BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """Return BM25 scores for *query* against the fitted corpus."""
        if self._bm25 is None:
            raise RuntimeError("BM25Encoder must be fit() before get_scores().")
        tokens = self._tokenize(query)
        return self._bm25.get_scores(tokens).astype(np.float32)


# Factory
def get_encoder(name: str, **kwargs):
    """
    Convenience factory.
    Parameters
    ----------
    name : ``'sbert'`` | ``'tfidf'`` | ``'bm25'``
    """
    name = name.lower()
    if name == "sbert":
        return SBERTEncoder(**kwargs)
    elif name == "tfidf":
        return TFIDFEncoder(**kwargs)
    elif name == "bm25":
        return BM25Encoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder: {name!r}. Choose sbert | tfidf | bm25.")

if __name__ == "__main__":
    enc = TFIDFEncoder(n_components=64)
    sentences = ["The cat sat on the mat.", "Dogs love to play fetch.", "Birds fly high."]
    vecs = enc.fit_encode(sentences)
    print("TF-IDF vectors shape:", vecs.shape)
