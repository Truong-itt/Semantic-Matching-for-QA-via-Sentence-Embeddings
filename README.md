# Semantic Matching for QA via Sentence Embeddings
## Unsupervised vs Supervised Sentence Selection

> **NLP Project** — Sentence Selection Question Answering on SQuAD v1.1

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Pipeline Overview](#5-pipeline-overview)
6. [Quickstart](#6-quickstart)
7. [Models](#7-models)
8. [Evaluation](#8-evaluation)
9. [Results (Expected)](#9-results-expected)
10. [Error Analysis](#10-error-analysis)
11. [Extending the Project](#11-extending-the-project)

---

## 1. Problem Statement

Standard **QA (Question Answering)** systems with SQuAD predict an exact **span** from a passage.
This project simplifies the task to **Sentence Selection**:

> Given a question **Q** and a context paragraph split into sentences **{S₁, S₂, …, Sₙ}**,
> select the sentence **Sᵢ** most likely to contain the answer.

### Input / Output

| Key        | Description |
|------------|-------------|
| `question` | Natural language question string |
| `sentences`| List of sentences from the passage |
| `label`    | Gold index — the sentence containing the answer span |

---

## 2. Dataset

**SQuAD v1.1** — Stanford Question Answering Dataset

| Split      | Samples (approx.) |
|------------|-------------------|
| Train      | 87 599            |
| Validation | 10 570            |

Loaded via HuggingFace `datasets`:
```python
from datasets import load_dataset
dataset = load_dataset("squad")
```

The dataset is converted from span-extraction to sentence-selection format
by the `preprocessing/convert_dataset.py` module.

---

## 3. Project Structure

```
sentence_selection_qa/
│
├── data/
│   ├── raw/                 ← Raw SQuAD files (auto-downloaded)
│   └── processed/           ← Converted sentence-selection JSON
│       ├── train.json
│       └── val.json
│
├── preprocessing/
│   ├── __init__.py
│   ├── sentence_tokenizer.py   ← NLTK / regex sentence splitter
│   └── convert_dataset.py      ← SQuAD → sentence-selection converter
│
├── embeddings/
│   ├── __init__.py
│   ├── bilstm_encoder.py        ← PyTorch BiLSTM sentence encoder
│   └── pretrained_encoder.py   ← TF-IDF/SVD, Sentence-BERT, BM25
│
├── models/
│   ├── __init__.py
│   ├── unsupervised.py     ← Cosine / Euclidean similarity selector
│   └── supervised.py       ← LR / RF / XGBoost binary classifier
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          ← Acc@k, MRR, P/R/F1, plots
│   └── error_analysis.py   ← Lexical bias, paraphrase, context length
│
├── experiments/
│   ├── __init__.py
│   ├── run_unsupervised.py ← Unsupervised experiment runner
│   └── run_supervised.py   ← Supervised experiment runner
│
├── results/
│   ├── tables/             ← CSV / JSON result files
│   └── plots/              ← PNG comparison charts
│
├── saved_models/           ← Pickled sklearn classifiers
│
├── main.py                 ← Unified CLI entry point
└── README.md               ← This file
```

---

## 4. Installation

### Prerequisites

- Python ≥ 3.10
- pip

### Install dependencies

```bash
pip install torch datasets nltk scikit-learn numpy matplotlib
```

### Optional (for SBERT and XGBoost experiments)

```bash
pip install sentence-transformers xgboost rank-bm25
```

### Download NLTK data

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

---

## 5. Pipeline Overview

```
SQuAD v1.1 (HuggingFace)
        │
        ▼
┌────────────────────────┐
│  convert_dataset.py    │  • Sentence tokenization
│                        │  • Locate answer sentence
│                        │  → data/processed/{train,val}.json
└───────────┬────────────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
 UNSUPERVISED   SUPERVISED
─────────────  ────────────
Encode Q & S   Build (Q,Sᵢ) pair features:
with TF-IDF /   - cosine_sim (scalar)
SBERT / BM25    - elem_prod  (D-dim)
     │          - abs_diff   (D-dim)
     ▼          - concat     (2D-dim)
Cosine sim /         │
Euclidean       Train LR / RF / XGB
distance             │
     │               ▼
     └───────────────┤
                     ▼
            ┌─────────────────┐
            │   Evaluation    │
            │  Acc@k / MRR    │
            │   P/R/F1        │
            │ Error Analysis  │
            └────────┬────────┘
                     ▼
              results/ tables + plots
```

---

## 6. Quickstart

### Full pipeline

```bash
cd sentence_selection_qa
python main.py --max_train 3000 --max_val 500
```

### Preprocess only

```bash
python main.py --mode preprocess --max_train 10000 --max_val 2000
```

### Unsupervised experiments only

```bash
python main.py --mode unsupervised --max_val 1000
```

### Supervised experiments only

```bash
python main.py --mode supervised --max_train 5000 --max_val 1000
```

### Generate comparison table from saved results

```bash
python main.py --mode compare
```

### Quick interactive demo

```bash
python main.py --mode demo \
  --question "Who wrote Hamlet?" \
  --context "William Shakespeare wrote Hamlet around 1600. \
             Romeo and Juliet is another famous Shakespeare play. \
             The Globe Theatre was built in 1599."
```

### Include Sentence-BERT (if installed)

```bash
python main.py --mode full --use_sbert --max_train 5000 --max_val 500
```

---

## 7. Models

### A — Unsupervised Baseline (`models/unsupervised.py`)

| System | Encoder | Similarity |
|--------|---------|------------|
| TF-IDF + Cosine | TF-IDF + SVD | Cosine |
| TF-IDF + Euclidean | TF-IDF + SVD | Euclidean |
| SBERT + Cosine | all-MiniLM-L6-v2 | Cosine |
| SBERT + Euclidean | all-MiniLM-L6-v2 | Euclidean |
| BM25 | Sparse TF | BM25 Okapi |

### B — BiLSTM Encoder (`embeddings/bilstm_encoder.py`)

- Random / GloVe word embeddings
- 2-layer BiLSTM
- Max-pool / Mean-pool / Last-state pooling
- Linear projection to fixed output size

### C — Supervised Models (`models/supervised.py`)

Feature vector per (Q, Sᵢ) pair:

| Feature | Dimension |
|---------|-----------|
| Cosine similarity | 1 |
| Element-wise product | D |
| Absolute difference | D |
| Concatenation [q; s] | 2D |

Classifiers:

| Model | Library |
|-------|---------|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |

---

## 8. Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy@k** | Gold sentence in top-k predictions |
| **Recall@k** | Same as Acc@k in single-label setting |
| **MRR** | Mean Reciprocal Rank |
| **Precision** | Fraction of top-1 predictions that are correct |
| **F1** | Harmonic mean of P and R |

### Error Analysis Categories

| Category | Description |
|----------|-------------|
| Lexical overlap bias | Wrong sentence shares more keywords with Q than gold |
| Semantic paraphrase | Gold sentence has low word overlap with Q (different wording) |
| Long context confusion | Error rate by number of sentences in context |

---

## 9. Results (Expected)

> Note: Actual results vary by train/val split size and random seed.
> Run the pipeline to obtain concrete numbers.

| System | Acc@1 | Acc@3 | Acc@5 | MRR |
|--------|-------|-------|-------|-----|
| TF-IDF + Cosine | ~0.55 | ~0.78 | ~0.87 | ~0.65 |
| TF-IDF + Euclidean | ~0.45 | ~0.70 | ~0.81 | ~0.56 |
| BM25 | ~0.58 | ~0.80 | ~0.89 | ~0.67 |
| SBERT + Cosine | ~0.72 | ~0.91 | ~0.95 | ~0.80 |
| TF-IDF + LR | ~0.62 | ~0.83 | ~0.91 | ~0.71 |
| TF-IDF + RF | ~0.60 | ~0.82 | ~0.90 | ~0.69 |
| SBERT + LR | ~0.76 | ~0.93 | ~0.97 | ~0.84 |

### Key Findings

1. **Cosine > Euclidean** — Cosine similarity consistently outperforms Euclidean
   distance, likely because question and sentence lengths differ significantly.

2. **SBERT > TF-IDF** — Pre-trained contextual embeddings capture semantic
   paraphrase relationships that TF-IDF misses.

3. **BM25 competitive** — Sparse BM25 is a strong baseline, especially for
   factoid questions with strong lexical overlap.

4. **Supervised > Unsupervised** — The binary classifier with pair features
   outperforms direct similarity scoring, especially at Acc@1.

5. **Top error type: Lexical bias** — ~40-50% of errors occur when a wrong
   sentence has higher word overlap with the question.

---

## 10. Error Analysis

Output files in `results/error_analysis/`:

- `*_error_report.json` — JSON summary of all error categories
- `*_context_error.png` — Error rate vs context length bar chart

Run standalone:
```python
from evaluation.error_analysis import run_error_analysis
run_error_analysis(selector, val_data, save_dir="results/error_analysis", tag="my_model")
```

---

## 11. Extending the Project

### Add a new encoder

1. Create a class with `encode(List[str]) -> np.ndarray` in `embeddings/`.
2. Pass it to `UnsupervisedSelector` or `SupervisedSelector`.

### Add a new classifier

1. Register it in `SupervisedSelector.MODEL_REGISTRY`.
2. Re-run `experiments/run_supervised.py`.

### Use full SQuAD

```bash
python main.py --mode full --max_train 87000 --max_val 10000 --use_sbert
```

### Extend to multi-sentence answer

Modify `preprocessing/convert_dataset.py` to allow `label` to be a list of integers
and update metrics accordingly.

---

## License

MIT License — for academic and research use.
