"""
Stage 4 - Training script for the digit model. Mirrors src/train.py.

Run from the project root:
    python -m digits.train
"""
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .dataset import DigitCommandDataset, scan_dataset, split_dataset_by_speaker
from .model import DigitCommandCNN


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = scan_dataset()
    if not samples:
        raise RuntimeError(f"No data found in {config.DATA_DIR}. Run prepare_digits_dataset.py and generate_silence.py first.")
    train_samples, val_samples, _ = split_dataset_by_speaker(samples)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} samples")

    train_ds = DigitCommandDataset(train_samples, augment=True)
    val_ds = DigitCommandDataset(val_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = DigitCommandCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = []
    best_val_acc = 0.0
    log_path = config.RESULTS_DIR / "train_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

        for epoch in range(1, config.EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            elapsed = time.time() - t0

            print(f"Epoch {epoch:02d}/{config.EPOCHS} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, elapsed])
            f.flush()
            history.append((epoch, train_loss, train_acc, val_loss, val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), config.CHECKPOINT_DIR / "best_model.pt")
                print(f"  -> new best model saved (val_acc={val_acc:.4f})")

    torch.save(model.state_dict(), config.CHECKPOINT_DIR / "last_model.pt")
    _plot_history(history)
    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Logs: {log_path}")


def _plot_history(history):
    epochs = [h[0] for h in history]
    train_loss = [h[1] for h in history]
    val_loss = [h[3] for h in history]
    train_acc = [h[2] for h in history]
    val_acc = [h[4] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "training_curves.png", dpi=150)
    print(f"Saved plot -> {config.RESULTS_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
