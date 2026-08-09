import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
CLASS_NAMES = ["Safe", "Warning", "Critical"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_sensor_columns(df):
    return [c for c in df.columns if c.startswith("sensor")]


def build_sequence_windows(df, kind, sensor_cols, window_size=WINDOW_SIZE):
    subset = df[df["kind"] == kind]
    X, y = [], []
    for _, engine_df in subset.groupby("sample"):
        engine_df = engine_df.sort_values("cycles")
        values = engine_df[sensor_cols].values  # (T, n_sensors)
        labels = engine_df["label"].values
        if len(engine_df) < window_size:
            continue
        for start in range(0, len(engine_df) - window_size + 1):
            window = values[start:start + window_size]              # (window, n_sensors)
            window = window - window.mean(axis=0, keepdims=True)    # denoise (remove DC/drift)
            spectrum = np.abs(np.fft.fft(window, axis=0))           # FFT magnitude spectrum
            X.append(spectrum.T)                                    # (n_sensors, window)
            y.append(labels[start + window_size - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class SensorWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    def __init__(self, n_sensors, n_classes=3, window_size=WINDOW_SIZE):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_sensors, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        reduced_len = window_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced_len, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)


def _run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total_count += X_batch.size(0)

    return total_loss / total_count, total_correct / total_count


def run(dataset_path, documents_dir):
    os.makedirs(documents_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    sensor_cols = _get_sensor_columns(df)
    n_sensors = len(sensor_cols)

    print("[CNN] Building sequence windows...")
    X_train, y_train = build_sequence_windows(df, "train", sensor_cols)
    X_val, y_val = build_sequence_windows(df, "validation", sensor_cols)
    X_test, y_test = build_sequence_windows(df, "test", sensor_cols)

    train_loader = DataLoader(SensorWindowDataset(X_train, y_train),
                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SensorWindowDataset(X_val, y_val),
                             batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SensorWindowDataset(X_test, y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = CNN1D(n_sensors=n_sensors).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("[CNN] Training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[CNN] Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # ---- Evaluate on test split -----------------------------------------------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    test_loss = history["val_loss"][-1]  # reference value; see also curve below

    print(f"[CNN] Test Accuracy={acc:.4f}  Precision={prec:.4f}  "
          f"Recall={rec:.4f}  F1={f1:.4f}")

    # ---- Loss curve --------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN - Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "cnn_loss_curve.png"), dpi=150)
    plt.close()

    # ---- Accuracy curve -----------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN - Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "cnn_accuracy_curve.png"), dpi=150)
    plt.close()

    # ---- Confusion matrix -----------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("CNN - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "cnn_confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- Metrics bar chart ------------------------------------------------
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    plt.figure(figsize=(5, 4))
    plt.bar(metrics.keys(), metrics.values(), color="#55A868")
    plt.ylim(0, 1)
    plt.title("CNN - Test Metrics")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(documents_dir, "cnn_metrics.png"), dpi=150)
    plt.close()

    # Save trained weights for re-use by CNN_FUZZY.py
    weights_path = os.path.join(documents_dir, "..", "..", "src", "cnn_weights.pt")
    torch.save(model.state_dict(), os.path.normpath(weights_path))

    return {"model": "1D-CNN", "accuracy": acc,
            "precision": prec, "recall": rec, "f1_score": f1}


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(
        dataset_path=os.path.join(base, "Dataset", "dataset.csv"),
        documents_dir=os.path.join(base, "Documents", "CNN_documents"),
    )
