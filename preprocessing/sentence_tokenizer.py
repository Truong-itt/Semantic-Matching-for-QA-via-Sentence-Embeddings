"""
sentence_tokenizer.py
---------------------
Utility for splitting a raw context paragraph into individual sentences.
Supports both rule-based (NLTK) and regex-based fallback tokenization.
"""

import re
from typing import List

try:
    import nltk
    # Download punkt tokenizer data silently if not present
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


def _regex_sent_tokenize(text: str) -> List[str]:
    """
    Simple regex-based sentence tokenizer as a fallback when NLTK is
    unavailable.  Splits on [.!?] followed by whitespace + uppercase letter.
    """
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize_sentences(text: str, method: str = "nltk") -> List[str]:
    """
    Split *text* into a list of sentences.

    Parameters
    ----------
    text   : str  –  Raw paragraph / context string.
    method : str  –  ``'nltk'`` (default) or ``'regex'``.

    Returns
    -------
    List[str]  –  Ordered list of sentences.
    """
    if method == "nltk" and _NLTK_AVAILABLE:
        sentences = _nltk_sent_tokenize(text)
    else:
        sentences = _regex_sent_tokenize(text)

    # Filter empty strings that may occur after splitting
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def find_answer_sentence_idx(sentences: List[str], answer_text: str) -> int:
    """
    Return the index of the *first* sentence that contains *answer_text*.
    Returns -1 if no sentence contains the answer.

    Parameters
    ----------
    sentences   : List[str]  –  Tokenised sentences from the context.
    answer_text : str        –  The answer span to locate.

    Returns
    -------
    int – Index of the answer sentence, or -1 if not found.
    """
    answer_lower = answer_text.lower()
    for idx, sent in enumerate(sentences):
        if answer_lower in sent.lower():
            return idx
    return -1


if __name__ == "__main__":
    sample_ctx = (
        "The Amazon rainforest is the world's largest tropical rainforest. "
        "It covers much of northwestern Brazil and extends into Colombia, "
        "Peru and other South American countries. "
        "The Amazon represents over half of the planet's remaining rainforests."
    )
    sents = tokenize_sentences(sample_ctx)
    for i, s in enumerate(sents):
        print(f"[{i}] {s}")
