"""
webapp/app.py
-------------
Flask web application for the Sentence Selection QA demo.

Run:
    python webapp/app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys
import json
import re

# ── Project root on path ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, render_template, request, jsonify
import numpy as np

from preprocessing.sentence_tokenizer import tokenize_sentences
from embeddings.pretrained_encoder    import TFIDFEncoder
from models.unsupervised              import UnsupervisedSelector

# ── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Global encoder (lazy-loaded on first request) ───────────────────────────
_encoder: TFIDFEncoder | None = None
_encoder_ready = False

# ── Sample SQuAD-style examples ─────────────────────────────────────────────
EXAMPLES = [
    {
        "title": "Amazon Rainforest",
        "question": "What is the largest tropical rainforest in the world?",
        "context": (
            "The Amazon rainforest is the world's largest tropical rainforest, "
            "covering much of northwestern Brazil and extending into Colombia, Peru "
            "and other South American countries. "
            "The Amazon represents over half of the planet's remaining rainforests "
            "and comprises the largest and most biodiverse tract of tropical rainforest "
            "in the world. "
            "Its electric eels can generate an electric shock of up to 600 volts. "
            "The forest is home to about 10% of all species on Earth. "
            "Deforestation in the Amazon has been a major environmental concern for decades."
        ),
    },
    {
        "title": "Isaac Newton",
        "question": "What famous work did Newton publish about universal gravitation?",
        "context": (
            "Isaac Newton formulated the law of universal gravitation in the 17th century. "
            "He published his landmark work Philosophiæ Naturalis Principia Mathematica in 1687, "
            "which described universal gravitation and the three laws of motion. "
            "Newton also made seminal contributions to optics and shares credit with "
            "Gottfried Wilhelm Leibniz for developing calculus. "
            "He was born in Woolsthorpe, Lincolnshire, England, on 4 January 1643. "
            "Newton served as Lucasian Professor of Mathematics at Cambridge University."
        ),
    },
    {
        "title": "World Wide Web",
        "question": "Who invented the World Wide Web?",
        "context": (
            "The World Wide Web was invented by British scientist Tim Berners-Lee in 1989. "
            "He was working at CERN, the European nuclear research organisation, at the time. "
            "The first website went live on 6 August 1991. "
            "The internet itself was developed in the late 1960s by ARPANET. "
            "Today over 5 billion people use the internet worldwide. "
            "Berners-Lee also founded the World Wide Web Consortium (W3C) to oversee "
            "the continued development of the Web."
        ),
    },
    {
        "title": "Python Language",
        "question": "Who created the Python programming language?",
        "context": (
            "Python is a high-level, general-purpose programming language. "
            "It was created by Guido van Rossum and first released in 1991. "
            "Python's design philosophy emphasises code readability. "
            "The Python Software Foundation manages the language. "
            "Python consistently ranks as one of the most popular programming languages. "
            "It is widely used in data science, machine learning, and web development."
        ),
    },
    {
        "title": "Mount Everest",
        "question": "What is the height of Mount Everest?",
        "context": (
            "Mount Everest is Earth's highest mountain above sea level, "
            "located in the Mahalangur Himal sub-range of the Himalayas. "
            "Its elevation of 8,848.86 metres was most recently established in 2020 "
            "by a Chinese survey. "
            "The first recorded ascent was by Edmund Hillary and Tenzing Norgay on 29 May 1953. "
            "The mountain is called Sagarmatha in Nepali and Chomolungma in Tibetan. "
            "Thousands of climbers have attempted to summit Everest since its first ascent."
        ),
    },
]


# ── Encoder helpers ──────────────────────────────────────────────────────────

def get_encoder() -> TFIDFEncoder:
    """Return a fitted TF-IDF encoder, training it the first time it's needed."""
    global _encoder, _encoder_ready
    if _encoder_ready and _encoder is not None:
        return _encoder

    # Fit on all example texts so the encoder works well for examples
    all_texts: list[str] = []
    for ex in EXAMPLES:
        all_texts.append(ex["question"])
        all_texts.extend(tokenize_sentences(ex["context"]))

    # Also load processed data if available
    proc_path = os.path.join(ROOT, "data", "processed", "val.json")
    if os.path.exists(proc_path):
        with open(proc_path, encoding="utf-8") as f:
            val_data = json.load(f)[:500]
        for s in val_data:
            all_texts.append(s["question"])
            all_texts.extend(s["sentences"])
        print(f"[Encoder] Fitting on {len(all_texts):,} texts (examples + SQuAD val) …")
    else:
        print(f"[Encoder] Fitting on {len(all_texts):,} texts (examples only) …")

    _encoder = TFIDFEncoder(n_components=256, max_features=50_000)
    _encoder.fit(all_texts)
    _encoder_ready = True
    print("[Encoder] Ready ✅")
    return _encoder


def predict_sentences(
    question: str,
    context:  str,
    metric:   str = "cosine",
) -> dict:
    """
    Core prediction: score all sentences in *context* for *question*.

    Returns a dict with sentences, scores, and the predicted index.
    """
    enc       = get_encoder()
    sentences = tokenize_sentences(context)
    if not sentences:
        return {"error": "No sentences found in context."}

    selector = UnsupervisedSelector(enc, metric=metric)
    scores   = selector.score(question, sentences).tolist()

    # Normalise scores to [0, 1] for display
    min_s, max_s = min(scores), max(scores)
    span = max_s - min_s if max_s != min_s else 1.0
    norm_scores = [(s - min_s) / span for s in scores]

    pred_idx = int(np.argmax(scores))

    return {
        "sentences"  : sentences,
        "raw_scores" : [round(s, 6) for s in scores],
        "norm_scores": [round(s, 4) for s in norm_scores],
        "predicted"  : pred_idx,
        "metric"     : metric,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", examples=EXAMPLES)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    context  = (data.get("context")  or "").strip()
    metric   = data.get("metric", "cosine").lower()

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not context:
        return jsonify({"error": "Context is required."}), 400
    if metric not in ("cosine", "euclidean"):
        metric = "cosine"

    result = predict_sentences(question, context, metric=metric)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Run both cosine and euclidean and return side-by-side."""
    data     = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    context  = (data.get("context")  or "").strip()

    if not question or not context:
        return jsonify({"error": "Question and context are required."}), 400

    cosine_res    = predict_sentences(question, context, metric="cosine")
    euclidean_res = predict_sentences(question, context, metric="euclidean")

    return jsonify({
        "cosine"   : cosine_res,
        "euclidean": euclidean_res,
    })


@app.route("/api/examples")
def api_examples():
    return jsonify(EXAMPLES)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Sentence Selection QA — Web Demo")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    # Warm-up the encoder before accepting requests
    get_encoder()
    app.run(host="0.0.0.0", port=5000, debug=False)
