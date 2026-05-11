"""
bilstm_encoder.py
-----------------
BiLSTM-based sentence encoder built with PyTorch.

Architecture
------------
1. Embedding layer  (random init or pre-trained GloVe/Word2Vec)
2. Bidirectional LSTM
3. Pooling layer    (max-pool or mean-pool over time steps)
4. Linear projection to a fixed-size output vector

Usage
-----
>>> vocab = {"[PAD]": 0, "the": 1, "cat": 2, ...}
>>> encoder = BiLSTMEncoder(vocab_size=len(vocab), embed_dim=100,
...                         hidden_dim=128, output_dim=256)
>>> # text_ids: LongTensor of shape (batch, seq_len)
>>> vecs = encoder(text_ids, lengths)   # (batch, 256)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Optional
import numpy as np


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM sentence encoder.

    Parameters
    ----------
    vocab_size  : Vocabulary size (including PAD token at index 0).
    embed_dim   : Word-embedding dimensionality.
    hidden_dim  : LSTM hidden-state size *per direction*.
    output_dim  : Final projected vector dimensionality.
    num_layers  : Number of stacked LSTM layers.
    dropout     : Dropout probability (applied between LSTM layers).
    pooling     : ``'max'`` | ``'mean'`` | ``'last'``.
    pad_idx     : Index of the padding token.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = "max",
        pad_idx: int = 0,
    ):
        super().__init__()

        self.pooling   = pooling
        self.pad_idx   = pad_idx
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM output dim = hidden_dim * 2  (forward + backward)
        self.dropout   = nn.Dropout(dropout)
        self.projector = nn.Linear(hidden_dim * 2, output_dim)
        self.activation = nn.Tanh()

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        lengths: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : LongTensor  (batch, seq_len)
        lengths   : LongTensor  (batch,)   actual sentence lengths (no padding)

        Returns
        -------
        Tensor (batch, output_dim)
        """
        embedded = self.dropout(self.embedding(input_ids))  # (B, L, E)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            outputs, _ = self.lstm(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # (B, L, 2H)
        else:
            outputs, _ = self.lstm(embedded)  # (B, L, 2H)

        if self.pooling == "max":
            # Replace padded positions with -inf before max-pooling
            if lengths is not None:
                mask = self._length_mask(lengths, outputs.size(1), outputs.device)
                outputs = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled, _ = outputs.max(dim=1)
        elif self.pooling == "mean":
            if lengths is not None:
                mask = self._length_mask(lengths, outputs.size(1), outputs.device)
                outputs = outputs * mask.unsqueeze(-1).float()
                pooled  = outputs.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
            else:
                pooled = outputs.mean(dim=1)
        elif self.pooling == "last":
            # Concatenate last forward and first backward hidden states
            pooled = torch.cat([outputs[:, -1, :outputs.size(-1)//2],
                                outputs[:, 0, outputs.size(-1)//2:]], dim=-1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        out = self.activation(self.projector(self.dropout(pooled)))  # (B, D)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _length_mask(
        lengths: torch.LongTensor, max_len: int, device: torch.device
    ) -> torch.BoolTensor:
        """Create a boolean mask from sequence lengths."""
        return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    # ------------------------------------------------------------------
    def load_pretrained_embeddings(
        self,
        word2vec: Dict[str, np.ndarray],
        vocab: Dict[str, int],
        freeze: bool = False,
    ):
        """
        Initialise the embedding layer with pre-trained vectors.

        Parameters
        ----------
        word2vec : dict  –  mapping  word  →  numpy vector.
        vocab    : dict  –  mapping  word  →  index.
        freeze   : bool  –  If True, embedding weights are not updated.
        """
        embed_matrix = self.embedding.weight.data
        found = 0
        for word, idx in vocab.items():
            if word in word2vec:
                embed_matrix[idx] = torch.tensor(word2vec[word], dtype=torch.float)
                found += 1
        print(f"Loaded {found}/{len(vocab)} pre-trained word vectors.")
        if freeze:
            self.embedding.weight.requires_grad = False


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

class SimpleVocab:
    """Build and manage a word vocabulary from a list of sentences."""

    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    def build(self, sentences: List[str]):
        from collections import Counter
        import re

        counter: Counter = Counter()
        for sent in sentences:
            tokens = re.findall(r"\b\w+\b", sent.lower())
            counter.update(tokens)

        self.word2idx = {self.PAD: 0, self.UNK: 1}
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        return self

    def encode(self, sentence: str, max_len: int = 64) -> List[int]:
        import re
        tokens = re.findall(r"\b\w+\b", sentence.lower())[:max_len]
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        return ids

    def __len__(self):
        return len(self.word2idx)


# ---------------------------------------------------------------------------
# Batch encoding helper
# ---------------------------------------------------------------------------

def encode_batch(
    sentences: List[str],
    vocab: SimpleVocab,
    max_len: int = 64,
    device: str = "cpu",
):
    """
    Encode a list of sentences into padded tensors.

    Returns
    -------
    ids     : LongTensor  (N, max_len)
    lengths : LongTensor  (N,)
    """
    encoded = [vocab.encode(s, max_len) for s in sentences]
    lengths = [len(e) for e in encoded]
    padded  = [e + [0] * (max_len - len(e)) for e in encoded]
    ids     = torch.tensor(padded, dtype=torch.long, device=device)
    lens    = torch.tensor(lengths, dtype=torch.long, device=device)
    return ids, lens
