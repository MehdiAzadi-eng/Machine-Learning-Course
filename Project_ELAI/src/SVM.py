import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

WINDOW_SIZE = 30
CLASS_NAMES = ["Safe", "Warning", "Critical"]


def _get_sensor_columns(df):
    return [c for c in df.columns if c.startswith("sensor")]


def _extract_window_features(window, sensor_cols):
    """Turn a (WINDOW_SIZE, n_sensors) slice into a 1D statistical feature vector."""
    feats = []
    for col in sensor_cols:
        values = window[col].values
        feats.extend(
            [
                values.mean(),
                values.std(),
                np.sqrt(np.mean(values ** 2)),          # RMS
                skew(values) if values.std() > 0 else 0.0,
                kurtosis(values) if values.std() > 0 else 0.0,
            ]
        )
    return feats


def build_feature_windows(df, kind, sensor_cols, window_size=WINDOW_SIZE):
    subset = df[df["kind"] == kind]
    X, y = [], []
    for _, engine_df in subset.groupby("sample"):
        engine_df = engine_df.sort_values("cycles")
        if len(engine_df) < window_size:
            continue
        for start in range(0, len(engine_df) - window_size + 1):
            window = engine_df.iloc[start:start + window_size]
            X.append(_extract_window_features(window, sensor_cols))
            y.append(window["label"].iloc[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def run(dataset_path, documents_dir):
    os.makedirs(documents_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    sensor_cols = _get_sensor_columns(df)

    print("[SVM] Building feature windows...")
    X_train, y_train = build_feature_windows(df, "train", sensor_cols)
    X_val, y_val = build_feature_windows(df, "validation", sensor_cols)
    X_test, y_test = build_feature_windows(df, "test", sensor_cols)

    # Merge train + validation: SVM has no internal validation loop
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    print("[SVM] Training SVC (RBF kernel)...")
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
    clf.fit(X_train_full, y_train_full)

    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"[SVM] Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}")

    # ---- Confusion matrix plot ----------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("SVM - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "svm_confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- Metrics bar chart ---------------------------------------------------
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    plt.figure(figsize=(5, 4))
    plt.bar(metrics.keys(), metrics.values(), color="#4C72B0")
    plt.ylim(0, 1)
    plt.title("SVM - Test Metrics")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "svm_metrics.png"), dpi=150)
    plt.close()

    return {"model": "SVM (baseline)", "accuracy": acc,
            "precision": prec, "recall": rec, "f1_score": f1}


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(
        dataset_path=os.path.join(base, "Dataset", "dataset.csv"),
        documents_dir=os.path.join(base, "Documents", "SVM_documents"),
    )
