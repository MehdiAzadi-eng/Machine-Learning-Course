"""
Main entry point.

Pipeline:
    1. Build Dataset/dataset.csv from the raw C-MAPSS FD001 text files
       (only if it does not already exist).
    2. Train & evaluate the 4 required models:
           - SVM (baseline)
           - 1D-CNN
           - LSTM
           - CNN + Fuzzy System
    3. Print / save a final comparison table
       (Accuracy, Loss, Precision, Recall, F1-Score).

Run with:
    python main.py
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
SRC_DIR = os.path.join(BASE_DIR, "src")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "Documents")
DATASET_PATH = os.path.join(DATASET_DIR, "dataset.csv")

sys.path.append(SRC_DIR)
sys.path.append(DATASET_DIR)


def ensure_dataset():
    if os.path.exists(DATASET_PATH):
        print(f"[main] Found existing dataset: {DATASET_PATH}")
        return

    required_raw_files = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    missing_raw = [f for f in required_raw_files
                   if not os.path.exists(os.path.join(DATASET_DIR, f))]

    print("=" * 78)
    print("[WARNING] Dataset/dataset.csv was not found.")
    if missing_raw:
        print("[WARNING] The following raw C-MAPSS files are also missing "
              f"from Dataset/: {missing_raw}")
        print("[WARNING] Download them (see the report, Section 2) and place "
              "them in the Dataset/ folder.")
    print("[WARNING] Then run:  python Dataset/dataset_normalization.py")
    print("[WARNING] ...before running main.py again.")
    print("=" * 78)
    raise FileNotFoundError(
        f"{DATASET_PATH} not found. Run Dataset/dataset_normalization.py first."
    )


def _fmt(x):
    """Format a metric value, showing N/A for models with no equivalent
    metric (e.g. SVM has no training 'loss' in the epoch/backprop sense)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    return f"{x:.4f}"


def print_comparison_table(results):
    df = pd.DataFrame(results)
    df = df[["model", "accuracy", "precision", "recall", "f1_score"]]
    df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]

    display_df = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = display_df[col].apply(_fmt)

    print("\n" + "=" * 78)
    print("FINAL MODEL COMPARISON - NASA C-MAPSS (FD001)")
    print("=" * 78)
    print(display_df.to_string(index=False))
    print("=" * 78)

    out_csv = os.path.join(BASE_DIR, "model_comparison_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"[main] Comparison table saved to {out_csv}")


def main():
    ensure_dataset()

    import src.SVM as SVM
    import src.CNN as CNN
    import src.LSTM as LSTM
    import src.CNN_FUZZY as CNN_FUZZY

    results = []

    print("\n[main] ===== Running SVM (baseline) =====")
    results.append(SVM.run(DATASET_PATH, os.path.join(DOCUMENTS_DIR, "SVM_documents")))

    print("\n[main] ===== Running 1D-CNN =====")
    results.append(CNN.run(DATASET_PATH, os.path.join(DOCUMENTS_DIR, "CNN_documents")))

    print("\n[main] ===== Running LSTM =====")
    results.append(LSTM.run(DATASET_PATH, os.path.join(DOCUMENTS_DIR, "LSTM_documents")))

    print("\n[main] ===== Running CNN + Fuzzy System =====")
    results.append(
        CNN_FUZZY.run(DATASET_PATH, os.path.join(DOCUMENTS_DIR, "CNN_FUZZY_documents"))
    )

    print_comparison_table(results)


if __name__ == "__main__":
    main()
