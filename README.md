# Semantic Matching for QA via Sentence Embeddings
## Unsupervised vs Supervised Sentence Selection

> NLP project: sentence-selection question answering on SQuAD v1.1.

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
9. [Results (Latest Run)](#9-results-latest-run)
10. [Error Analysis](#10-error-analysis)
11. [Extending the Project](#11-extending-the-project)

---

## 1. Problem Statement

Standard QA systems on SQuAD predict an exact span from a passage. This project
reframes the task as sentence selection:

> Given a question Q and a context paragraph split into sentences {S1, S2, ..., Sn},
> select the sentence Si most likely to contain the answer.

### Input / Output

| Key | Description |
|-----|-------------|
| `question` | Natural language question string |
| `sentences` | List of sentences from the passage |
| `label` | Gold index of the sentence containing the answer span |

---

## 2. Dataset

The project uses **SQuAD v1.1** from HuggingFace `datasets`:

```python
from datasets import load_dataset
dataset = load_dataset("squad")
```

The raw span-extraction data is converted to sentence-selection format by
`preprocessing/convert_dataset.py` and stored in `data/processed/` when the
processed files are available.

The default pipeline reads the processed local files first, then slices them by
`--max_train` and `--max_val`.

---

## 3. Project Structure

```
sentence_selection_qa/
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- preprocessing/
|   |-- sentence_tokenizer.py
|   `-- convert_dataset.py
|
|-- embeddings/
|   |-- bilstm_encoder.py
|   `-- pretrained_encoder.py
|
|-- models/
|   |-- unsupervised.py
|   `-- supervised.py
|
|-- evaluation/
|   |-- metrics.py
|   `-- error_analysis.py
|
|-- experiments/
|   |-- run_unsupervised.py
|   `-- run_supervised.py
|
|-- results/
|   |-- tables/
|   `-- plots/
|
|-- saved_models/
|-- main.py
`-- README.md
```

---

## 4. Installation

### Prerequisites

- Python 3.10 or newer
- pip

### Install dependencies

```bash
pip install torch datasets nltk scikit-learn numpy matplotlib
```

### Optional packages

```bash
pip install sentence-transformers xgboost rank-bm25
```

### Download NLTK data

```python
import nltk
nltk.download("punkt")
```

---

## 5. Pipeline Overview

```
SQuAD v1.1
    |
    v
convert_dataset.py
    |-- sentence tokenization
    |-- locate answer sentence
    v
data/processed/{train,val}.json
    |
    +--------------------+
    |                    |
    v                    v
Unsupervised        Supervised
TF-IDF / SBERT /    (Q, Si) features -> LR / RF / XGB
BM25 encoders
    |                    |
    v                    v
Similarity score     Classification score
    \                  /
     \                /
      v              v
        Evaluation: Acc@k, MRR, P/R/F1
                |
                v
         results/tables + plots
```

---

## 6. Quickstart

### Full pipeline

```bash
cd sentence_selection_qa
python main.py
```

The default `full` mode runs preprocessing, unsupervised experiments,
supervised experiments, and the combined comparison table.

### Preprocess only

```bash
python main.py --mode preprocess --max_train 10000 --max_val 2000
```

### Unsupervised experiments only

```bash
python main.py --mode unsupervised --max_val 1000
```

This runner evaluates TF-IDF + Cosine, TF-IDF + Euclidean, BM25, and SBERT if
`sentence-transformers` is installed.

### Supervised experiments only

```bash
python main.py --mode supervised --max_train 5000 --max_val 1000
```

This runner trains TF-IDF + LR/RF/XGB, SBERT + LR/RF, and adds the TF-IDF +
Cosine unsupervised baseline for comparison.

### Compare saved results

```bash
python main.py --mode compare
```

### Demo mode

```bash
python main.py --mode demo \
  --question "Who wrote Hamlet?" \
  --context "William Shakespeare wrote Hamlet around 1600. Romeo and Juliet is another famous Shakespeare play. The Globe Theatre was built in 1599."
```

### Include Sentence-BERT

```bash
python main.py --mode full --use_sbert --max_train 5000 --max_val 500
```

Note: the current experiment runners already try SBERT when the dependency is
available, so `--use_sbert` is kept only for backward compatibility in the CLI.

---

## 7. Models

### A. Unsupervised baseline (`models/unsupervised.py`)

| System | Encoder | Similarity |
|--------|---------|------------|
| TF-IDF + Cosine | TF-IDF + SVD | Cosine |
| TF-IDF + Euclidean | TF-IDF + SVD | Euclidean |
| SBERT + Cosine | all-MiniLM-L6-v2 | Cosine |
| SBERT + Euclidean | all-MiniLM-L6-v2 | Euclidean |
| BM25 | Sparse TF | BM25 Okapi |

The unsupervised runner evaluates TF-IDF and BM25 by default, then attempts
SBERT and skips it gracefully if `sentence-transformers` is missing.

### B. BiLSTM encoder (`embeddings/bilstm_encoder.py`)

- Random or GloVe word embeddings
- 2-layer BiLSTM
- Max-pool, mean-pool, or last-state pooling
- Linear projection to a fixed output size

BiLSTM is implemented as a reusable encoder, but it is not currently wired into
the main experiment runners.

### C. Supervised models (`models/supervised.py`)

Feature vector per (Q, Si) pair:

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

The supervised runner uses TF-IDF features by default, then trains SBERT-based
variants when the dependency is installed.

---

## 8. Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy@k | Gold sentence appears in the top-k predictions |
| Recall@k | Same as Acc@k in this single-label setup |
| MRR | Mean Reciprocal Rank |
| Precision | Fraction of top-1 predictions that are correct |
| F1 | Harmonic mean of precision and recall |

### Error analysis categories

| Category | Description |
|----------|-------------|
| Lexical overlap bias | Wrong sentence shares more keywords with the question than the gold sentence |
| Semantic paraphrase | Gold sentence has low word overlap with the question |
| Long context confusion | Error rate by number of sentences in the context |

The code writes error analysis artifacts to `results/error_analysis/` using tags
such as `tfidf_cosine` and `tfidf_lr`.

---

## 9. Results (Latest Run)

Latest saved results were produced with `python main.py --mode compare` after
training with `--max_train 5000 --max_val 1000`.
See `results/tables/unsupervised_results.json` and
`results/tables/supervised_results.json` for the canonical metric sources.

| System | Acc@1 | Acc@3 | Acc@5 | MRR | f1 |
|--------|-------|-------|-------|-----|----|
| TF-IDF + Cosine | 0.7244 | 0.9439 | 0.9850 | 0.8338 | 0.7244 |
| TF-IDF + Euclidean | 0.7244 | 0.9439 | 0.9850 | 0.8338 | 0.7244 |
| BM25 | 0.7214 | 0.9539 | 0.9850 | 0.8393 | 0.7214 |
| SBERT + Cosine | 0.7615 | 0.9599 | 0.9870 | 0.8594 | 0.7615 |
| SBERT + Euclidean | 0.7615 | 0.9599 | 0.9870 | 0.8594 | 0.7615 |
| TF-IDF + LR | 0.6964 | 0.9259 | 0.9760 | 0.8116 | 0.6964 |
| TF-IDF + RF | 0.7335 | 0.9369 | 0.9850 | 0.8356 | 0.7335 |
| TF-IDF + XGB | 0.7194 | 0.9489 | 0.9870 | 0.8337 | 0.7194 |
| SBERT + LR | 0.7555 | 0.9599 | 0.9830 | 0.8550 | 0.7555 |
| SBERT + RF | 0.7565 | 0.9539 | 0.9890 | 0.8568 | 0.7565 |
| TF-IDF + Cosine (Unsup) | 0.7244 | 0.9439 | 0.9850 | 0.8338 | 0.7244 |
| tfidf_LR_neg1 | 0.7034 | 0.9319 | 0.9770 | 0.8154 | 0.7034 |
| tfidf_LR_neg3 | 0.6964 | 0.9259 | 0.9760 | 0.8116 | 0.6964 |
| tfidf_LR_neg5 | 0.6954 | 0.9309 | 0.9780 | 0.8105 | 0.6954 |

The compare mode also regenerates `results/tables/all_results.csv` and the
plots in `results/plots/`.

### Key findings

1. SBERT is the strongest encoder and is more robust to paraphrase than TF-IDF.
2. SBERT + Cosine is the best overall model by Acc@1, while SBERT + RF is the
   best supervised model.
3. BM25 remains a strong baseline for factoid-style questions.
4. TF-IDF + Euclidean is effectively tied with cosine in the current setup,
   because both operate on normalized TF-IDF/SVD vectors.
5. The dominant failure mode is lexical overlap bias.

---

## 10. Error Analysis

Output files in `results/error_analysis/`:

- `*_error_report.json` - JSON summary of the error categories
- `*_context_error.png` - Error rate vs. context length chart

The current runner creates reports for the best unsupervised TF-IDF cosine
model and the best supervised TF-IDF LR model.

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
python main.py --mode full --max_train 87000 --max_val 10000
```

If you want a thesis-ready write-up, see `docs/BaoCao_Day_Du.md` for the
expanded report with methodology, tables, and real experimental numbers.

### Extend to multi-sentence answer

Modify `preprocessing/convert_dataset.py` to allow `label` to be a list of
integers and update metrics accordingly.

---

## License

MIT License - for academic and research use.