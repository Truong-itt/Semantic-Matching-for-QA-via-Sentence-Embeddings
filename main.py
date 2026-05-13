"""
main.py
-------
Unified entry point for the Sentence Selection QA pipeline.

Usage
-----
python main.py                          # full pipeline (default settings)
python main.py --mode unsupervised      # unsupervised only
python main.py --mode supervised        # supervised only
python main.py --mode compare           # side-by-side comparison table
python main.py --max_train 1000 --max_val 200   # quick demo
python main.py --use_sbert              # include Sentence-BERT encoders
python main.py --mode demo --question "Who discovered gravity?" \
               --context "Newton discovered gravity. Archimedes discovered buoyancy."
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Argument parsing
def parse_args():
    p = argparse.ArgumentParser(
        description="Sentence Selection QA – main pipeline runner"
    )
    p.add_argument(
        "--mode",
        choices=["full", "preprocess", "unsupervised", "supervised", "compare", "demo"],
        default="full",
        help="Pipeline mode to run (default: full)",
    )
    p.add_argument("--max_train",  type=int,  default=5000, help="Max training samples to use (default: 5000)")
    p.add_argument("--max_val",    type=int,  default=1000, help="Max validation samples to use (default: 1000)")
    p.add_argument("--use_sbert",  action="store_true", help="Include Sentence-BERT encoder experiments")
    p.add_argument("--no_error_analysis", action="store_true",
                   help="Skip error analysis (faster)")
    p.add_argument("--no_ablation", action="store_true",
                   help="Skip feature ablation (faster)")
    p.add_argument("--neg_per_pos", type=int, default=3,
                   help="Negative examples per positive for supervised training")
    p.add_argument("--question", type=str, default=None, help="Demo question")
    p.add_argument("--context",  type=str, default=None, help="Demo context paragraph")
    return p.parse_args()

# Mode: preprocess
def mode_preprocess(args):
    from preprocessing.convert_dataset import load_and_convert
    print("STEP 1 — Data Preprocessing")
    load_and_convert(max_train=args.max_train, max_val=args.max_val)

# Mode: unsupervised
def mode_unsupervised(args):
    from experiments.run_unsupervised import run_unsupervised_experiments
    print("STEP 2 — Unsupervised Experiments")
    return run_unsupervised_experiments(max_train = args.max_train, max_val = args.max_val, run_error_analysis_flag= not args.no_error_analysis,)

# Mode: supervised
def mode_supervised(args):
    from experiments.run_supervised import run_supervised_experiments
    print("=" * 60)
    print("STEP 3 — Supervised Experiments")
    print("=" * 60)
    return run_supervised_experiments(
        max_train              = args.max_train,
        max_val                = args.max_val,
        neg_per_pos            = args.neg_per_pos,
        run_ablation           = not args.no_ablation,
        run_error_analysis_flag= not args.no_error_analysis,
    )

# Mode: compare
def mode_compare(args):
    """Load saved result JSON files and print a combined comparison table."""
    from evaluation.metrics import (
        format_results_table, save_results_csv, plot_acc_at_k, plot_mrr,
    )

    tables_dir  = os.path.join(ROOT, "results", "tables")
    plots_dir   = os.path.join(ROOT, "results", "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    combined: dict = {}
    for fname in ("unsupervised_results.json", "supervised_results.json"):
        fpath = os.path.join(tables_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                combined.update(json.load(f))
        else:
            print(f"Warning: {fpath} not found – run --mode unsupervised / supervised first.")

    if not combined:
        print("No results found. Run the pipeline first.")
        return

    print("\n" + "=" * 80)
    print("FULL COMPARISON TABLE")
    print("=" * 80)
    display = ["Accuracy@1", "Accuracy@3", "Accuracy@5", "MRR", "f1"]
    print(format_results_table(combined, metrics=display))

    # Save combined CSV
    save_results_csv(combined, os.path.join(tables_dir, "all_results.csv"))
    # Combined plots
    plot_acc_at_k(
        combined,
        k_values=(1, 3, 5),
        save_path=os.path.join(plots_dir, "all_acc_at_k.png"),
        title="All Systems – Accuracy@k",
    )
    plot_mrr(
        combined,
        save_path=os.path.join(plots_dir, "all_mrr.png"),
        title="All Systems – MRR",
    )

# Mode: demo
def mode_demo(args):
    """Interactive demo: predict the answer sentence for a given (Q, context)."""
    from preprocessing.sentence_tokenizer import tokenize_sentences
    from embeddings.pretrained_encoder    import TFIDFEncoder
    from models.unsupervised              import UnsupervisedSelector

    # Default demo if no args provided
    question = args.question or (
        "Who discovered the law of universal gravitation?"
    )
    context = args.context or (
        "Isaac Newton formulated the law of universal gravitation in 1687. "
        "Albert Einstein later revisited gravity with his theory of general relativity. "
        "Galileo Galilei performed early experiments on falling bodies. "
        "Newton's work was published in his famous Principia Mathematica."
    )

    print("\n" + "=" * 60)
    print("DEMO – Sentence Selection QA")
    print("=" * 60)
    print(f"Question : {question}")
    print(f"Context  : {context}\n")

    sentences = tokenize_sentences(context)
    print("Sentences:")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")

    # Fit a tiny TF-IDF encoder on just this context + question
    enc = TFIDFEncoder(n_components=min(32, len(sentences) + 1))
    all_texts = [question] + sentences
    enc.fit(all_texts)
    selector = UnsupervisedSelector(enc, metric="cosine")
    scores   = selector.score(question, sentences)
    print("\nScores (cosine similarity):")
    for i, (s, sc) in enumerate(zip(sentences, scores)):
        marker = " ◄ SELECTED" if i == int(scores.argmax()) else ""
        print(f"  [{i}] {sc:+.4f}  {s}{marker}")
    pred = selector.predict(question, sentences)
    print(f"\nSelected sentence [{pred}]: {sentences[pred]}")

# Main dispatcher
def main():
    args = parse_args()
    print("\n" + "#" * 70)
    print("#  Semantic Matching for QA via Sentence Embeddings")
    print("#  Mode: " + args.mode.upper())
    print("#" * 70 + "\n")
    if args.mode == "preprocess":
        mode_preprocess(args)
    elif args.mode == "unsupervised":
        mode_unsupervised(args)
    elif args.mode == "supervised":
        mode_supervised(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "demo":
        mode_demo(args)
    elif args.mode == "full":
        mode_preprocess(args)
        unsup_res = mode_unsupervised(args)
        sup_res   = mode_supervised(args)
        mode_compare(args)
    print("\nDone.")

if __name__ == "__main__":
    main()
