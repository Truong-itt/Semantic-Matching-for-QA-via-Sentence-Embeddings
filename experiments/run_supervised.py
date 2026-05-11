"""
run_supervised.py
-----------------
End-to-end experiment runner for supervised sentence-selection models.

Experiments
-----------
1. TF-IDF features + Logistic Regression
2. TF-IDF features + Random Forest
3. TF-IDF features + XGBoost          (if xgboost is installed)
4. SBERT features  + Logistic Regression  (if sentence-transformers installed)
5. SBERT features  + Random Forest

Comparison table: Supervised vs best Unsupervised is printed at the end.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
from typing import Dict, List, Any

from preprocessing.convert_dataset import load_processed, load_and_convert
from embeddings.pretrained_encoder  import TFIDFEncoder, SBERTEncoder
from models.supervised               import SupervisedSelector
from models.unsupervised             import UnsupervisedSelector
from evaluation.metrics              import (
    evaluate_selector,
    format_results_table,
    save_results_csv,
    plot_acc_at_k,
    plot_mrr,
    plot_confusion_rank_distribution,
)
from evaluation.error_analysis import run_error_analysis


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR  = os.path.join(ROOT, "results")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR   = os.path.join(ROOT, "saved_models")
K_VALUES     = (1, 3, 5)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_data(max_train: int = 5000, max_val: int = 1000):
    proc_train = os.path.join(ROOT, "data", "processed", "train.json")
    proc_val   = os.path.join(ROOT, "data", "processed", "val.json")

    if os.path.exists(proc_train) and os.path.exists(proc_val):
        train = load_processed("train")[:max_train]
        val   = load_processed("val")[:max_val]
    else:
        train, val = load_and_convert(save=True, max_train=max_train, max_val=max_val)

    print(f"Train: {len(train)}  |  Val: {len(val)}")
    return train, val


def build_tfidf(train_data, val_data) -> TFIDFEncoder:
    texts = []
    for s in train_data + val_data:
        texts.append(s["question"])
        texts.extend(s["sentences"])
    enc = TFIDFEncoder(n_components=256)
    print("Fitting TF-IDF encoder …")
    enc.fit(texts)
    return enc


def build_sbert() -> SBERTEncoder:
    return SBERTEncoder(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Feature ablation experiment
# ---------------------------------------------------------------------------

def run_feature_ablation(
    train_data: List[Dict],
    val_data:   List[Dict],
    encoder,
    tag_prefix: str = "tfidf",
) -> Dict[str, Dict]:
    """
    Train Logistic Regression with different feature subsets to ablate.
    Here we vary neg_per_pos ratio which affects class balance.
    """
    from models.supervised import SupervisedSelector, build_dataset
    ablation_results: Dict[str, Dict] = {}

    for neg_ratio in (1, 3, 5):
        name = f"{tag_prefix}_LR_neg{neg_ratio}"
        print(f"\n  Ablation: {name}")
        sel = SupervisedSelector(model_type="lr", encoder=encoder)
        sel.train(train_data, neg_per_pos=neg_ratio)
        res = evaluate_selector(sel, val_data, top_k_values=K_VALUES)
        ablation_results[name] = res

    return ablation_results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_supervised_experiments(
    max_train: int = 5000,
    max_val:   int = 1000,
    use_sbert: bool = False,
    neg_per_pos: int = 3,
    run_ablation: bool = True,
    run_error_analysis_flag: bool = True,
    save_models: bool = True,
):
    os.makedirs(TABLES_DIR,  exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    # ---- Data
    train_data, val_data = get_data(max_train, max_val)

    # ---- Encoder
    tfidf_enc = build_tfidf(train_data, val_data)

    all_results: Dict[str, Dict] = {}

    # ---- TF-IDF feature experiments
    for model_type in ("lr", "rf"):
        name = f"TF-IDF + {model_type.upper()}"
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        sel = SupervisedSelector(model_type=model_type, encoder=tfidf_enc)
        sel.train(train_data, neg_per_pos=neg_per_pos)
        res = evaluate_selector(sel, val_data, top_k_values=K_VALUES)
        all_results[name] = res

        if save_models:
            sel.save(os.path.join(MODELS_DIR, f"tfidf_{model_type}.pkl"))

    # XGBoost (optional)
    try:
        import xgboost  # noqa
        name = "TF-IDF + XGB"
        print(f"\nTraining: {name}")
        sel = SupervisedSelector(model_type="xgb", encoder=tfidf_enc)
        sel.train(train_data, neg_per_pos=neg_per_pos)
        res = evaluate_selector(sel, val_data, top_k_values=K_VALUES)
        all_results[name] = res
        if save_models:
            sel.save(os.path.join(MODELS_DIR, "tfidf_xgb.pkl"))
    except ImportError:
        print("XGBoost not installed – skipping XGB experiment.")

    # ---- SBERT feature experiments (optional)
    if use_sbert:
        try:
            sbert_enc = build_sbert()
            for model_type in ("lr", "rf"):
                name = f"SBERT + {model_type.upper()}"
                print(f"\nTraining: {name}")
                sel = SupervisedSelector(model_type=model_type, encoder=sbert_enc)
                sel.train(train_data, neg_per_pos=neg_per_pos)
                res = evaluate_selector(sel, val_data, top_k_values=K_VALUES)
                all_results[name] = res
                if save_models:
                    sel.save(os.path.join(MODELS_DIR, f"sbert_{model_type}.pkl"))
        except ImportError:
            print("sentence-transformers not installed – skipping SBERT supervised.")

    # ---- Add unsupervised baseline for comparison
    print("\nAdding unsupervised TF-IDF + Cosine baseline for comparison …")
    unsup_sel = UnsupervisedSelector(tfidf_enc, metric="cosine")
    all_results["TF-IDF + Cosine (Unsup)"] = evaluate_selector(unsup_sel, val_data, K_VALUES)

    # ---- Feature ablation
    if run_ablation:
        print("\n\nRunning feature ablation …")
        ablation_res = run_feature_ablation(train_data, val_data, tfidf_enc, "tfidf")
        all_results.update(ablation_res)

    # ---- Print table
    print("\n" + "=" * 80)
    print("SUPERVISED EXPERIMENT RESULTS")
    print("=" * 80)
    display_metrics = [f"Accuracy@{k}" for k in K_VALUES] + ["MRR", "f1"]
    print(format_results_table(all_results, metrics=display_metrics))

    # ---- Save
    csv_path  = os.path.join(TABLES_DIR, "supervised_results.csv")
    json_path = os.path.join(TABLES_DIR, "supervised_results.json")
    save_results_csv(all_results, csv_path)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results saved → {json_path}")

    # ---- Plots
    # Use only primary models for clean plots (exclude ablation)
    primary_results = {k: v for k, v in all_results.items()
                       if "neg" not in k}

    plot_acc_at_k(
        primary_results,
        k_values=K_VALUES,
        save_path=os.path.join(PLOTS_DIR, "supervised_acc_at_k.png"),
        title="Supervised vs Unsupervised – Accuracy@k",
    )
    plot_mrr(
        primary_results,
        save_path=os.path.join(PLOTS_DIR, "supervised_mrr.png"),
        title="Supervised vs Unsupervised – MRR",
    )

    if run_ablation:
        ablation_primary = {k: v for k, v in all_results.items() if "neg" in k}
        if ablation_primary:
            plot_acc_at_k(
                ablation_primary,
                k_values=K_VALUES,
                save_path=os.path.join(PLOTS_DIR, "ablation_neg_ratio.png"),
                title="Feature Ablation – neg_per_pos ratio",
            )

    # ---- Error analysis on best supervised model
    if run_error_analysis_flag and "TF-IDF + LR" in all_results:
        best_lr = SupervisedSelector(model_type="lr", encoder=tfidf_enc)
        best_lr.train(train_data, neg_per_pos=neg_per_pos)
        run_error_analysis(
            best_lr,
            val_data,
            save_dir=os.path.join(RESULTS_DIR, "error_analysis"),
            tag="tfidf_lr",
        )

    return all_results


if __name__ == "__main__":
    run_supervised_experiments(
        max_train=3000,
        max_val=500,
        use_sbert=False,
        run_ablation=True,
        run_error_analysis_flag=True,
    )
