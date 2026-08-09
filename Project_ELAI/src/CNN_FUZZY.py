"""
Hybrid Neuro-Fuzzy model: 1D-CNN (feature extractor) + Mamdani Fuzzy
Inference System (decision layer).

Outputs saved to Documents/CNN_FUZZY_documents:
    - Fuzzy membership function plots (inputs + output)
    - Confusion matrix on the test split
    - Bar chart of Accuracy / Precision / Recall / F1-Score
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import skfuzzy as fuzz
from skfuzzy import control as ctrl
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from CNN import (  # noqa: E402  (import after sys.path tweak, on purpose)
    CNN1D,
    SensorWindowDataset,
    build_sequence_windows,
    _get_sensor_columns,
    WINDOW_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    _run_epoch,
)

CLASS_NAMES = ["Safe", "Warning", "Critical"]


# ------------------------------------------------------------------------
# Fuzzy Inference System
# ------------------------------------------------------------------------
def build_fuzzy_system():
    fault_probability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "fault_probability")
    rate_of_change = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "rate_of_change")
    risk = ctrl.Consequent(np.arange(0, 100.01, 1), "risk")

    fault_probability["low"] = fuzz.trimf(fault_probability.universe, [0, 0, 0.5])
    fault_probability["medium"] = fuzz.trimf(fault_probability.universe, [0.2, 0.5, 0.8])
    fault_probability["high"] = fuzz.trimf(fault_probability.universe, [0.5, 1, 1])

    rate_of_change["low"] = fuzz.trimf(rate_of_change.universe, [0, 0, 0.4])
    rate_of_change["medium"] = fuzz.trimf(rate_of_change.universe, [0.15, 0.4, 0.65])
    rate_of_change["high"] = fuzz.trimf(rate_of_change.universe, [0.4, 1, 1])

    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 45])
    risk["medium"] = fuzz.trimf(risk.universe, [25, 50, 75])
    risk["high"] = fuzz.trimf(risk.universe, [55, 100, 100])

    rules = [
        ctrl.Rule(fault_probability["low"] & rate_of_change["low"], risk["low"]),
        ctrl.Rule(fault_probability["low"] & rate_of_change["medium"], risk["low"]),
        ctrl.Rule(fault_probability["low"] & rate_of_change["high"], risk["medium"]),
        ctrl.Rule(fault_probability["medium"] & rate_of_change["low"], risk["medium"]),
        ctrl.Rule(fault_probability["medium"] & rate_of_change["medium"], risk["medium"]),
        ctrl.Rule(fault_probability["medium"] & rate_of_change["high"], risk["high"]),
        ctrl.Rule(fault_probability["high"] & rate_of_change["low"], risk["medium"]),
        ctrl.Rule(fault_probability["high"] & rate_of_change["medium"], risk["high"]),
        ctrl.Rule(fault_probability["high"] & rate_of_change["high"], risk["high"]),
    ]

    system = ctrl.ControlSystem(rules)
    return system, (fault_probability, rate_of_change, risk)


def risk_to_label(risk_score):
    if risk_score < 33:
        return 0    # Safe
    elif risk_score < 66:
        return 1    # Warning
    else:
        return 2    # Critical


def run(dataset_path, documents_dir):
    os.makedirs(documents_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    sensor_cols = _get_sensor_columns(df)
    n_sensors = len(sensor_cols)

    print("[CNN_FUZZY] Building sequence windows...")
    X_train, y_train = build_sequence_windows(df, "train", sensor_cols)
    X_val, y_val = build_sequence_windows(df, "validation", sensor_cols)

    train_loader = DataLoader(SensorWindowDataset(X_train, y_train),
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SensorWindowDataset(X_val, y_val),
                             batch_size=BATCH_SIZE, shuffle=False)

    # ---- Train the CNN feature extractor (same architecture as CNN.py) ------
    model = CNN1D(n_sensors=n_sensors).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("[CNN_FUZZY] Training CNN feature extractor...")
    final_val_loss = None
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)
        final_val_loss = val_loss
        print(f"[CNN_FUZZY] Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    # NOTE: this is the CNN feature-extractor's cross-entropy loss only.
    # The fuzzy decision layer itself has no trainable loss (its rules and
    # membership functions are fixed, not learned via backpropagation).

    # ---- Build the fuzzy system ------------------------------------------
    fuzzy_system, (fp_var, roc_var, risk_var) = build_fuzzy_system()

    # ---- Run inference engine-by-engine so rate_of_change is meaningful ----
    model.eval()
    test_df = df[df["kind"] == "test"]

    all_preds, all_labels, all_risk = [], [], []
    with torch.no_grad():
        for _, engine_df in test_df.groupby("sample"):
            engine_df = engine_df.sort_values("cycles")
            values = engine_df[sensor_cols].values
            labels = engine_df["label"].values
            if len(engine_df) < WINDOW_SIZE:
                continue

            prev_prob = None
            for start in range(0, len(engine_df) - WINDOW_SIZE + 1):
                window = values[start:start + WINDOW_SIZE].T  # (n_sensors, window)
                x = torch.from_numpy(window).unsqueeze(0).float().to(DEVICE)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                critical_prob = float(probs[2])  # class index 2 = Critical

                rate = 0.0 if prev_prob is None else abs(critical_prob - prev_prob)
                prev_prob = critical_prob

                sim = ctrl.ControlSystemSimulation(fuzzy_system)
                sim.input["fault_probability"] = float(np.clip(critical_prob, 0, 1))
                sim.input["rate_of_change"] = float(np.clip(rate, 0, 1))
                sim.compute()
                risk_score = sim.output["risk"]

                all_risk.append(risk_score)
                all_preds.append(risk_to_label(risk_score))
                all_labels.append(labels[start + WINDOW_SIZE - 1])

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"[CNN_FUZZY] Test Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}")

    # ---- Membership function plots -----------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    for ax, var, name in zip(
        axes, [fp_var, roc_var, risk_var],
        ["Fault Probability", "Rate of Change", "Risk"]
    ):
        for label in var.terms:
            ax.plot(var.universe, var[label].mf, label=label)
        ax.set_title(name)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "fuzzy_membership_functions.png"), dpi=150)
    plt.close()

    # ---- Confusion matrix -----------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("CNN + Fuzzy System - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "cnn_fuzzy_confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- Metrics bar chart ------------------------------------------------
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    plt.figure(figsize=(5, 4))
    plt.bar(metrics.keys(), metrics.values(), color="#8172B2")
    plt.ylim(0, 1)
    plt.title("CNN + Fuzzy System - Test Metrics")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "cnn_fuzzy_metrics.png"), dpi=150)
    plt.close()

    return {"model": "CNN + Fuzzy System", "accuracy": acc,
            "precision": prec, "recall": rec, "f1_score": f1}


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(
        dataset_path=os.path.join(base, "Dataset", "dataset.csv"),
        documents_dir=os.path.join(base, "Documents", "CNN_FUZZY_documents"),
    )
