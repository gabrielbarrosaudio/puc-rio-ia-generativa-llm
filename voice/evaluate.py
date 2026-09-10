"""
Stage 4b - Evaluation script. Same as src/evaluate.py and digits/evaluate.py.

Run from the project root:
    python -m voice.evaluate
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .dataset import VoiceCommandDataset, scan_dataset, split_dataset_by_speaker
from .model import VoiceCommandCNN


def main():
    device = torch.device("cpu")
    model = VoiceCommandCNN().to(device)
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"No checkpoint at {ckpt_path}. Run train.py first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    samples = scan_dataset()
    _, _, test_samples = split_dataset_by_speaker(samples)
    if not test_samples:
        raise RuntimeError("Test split is empty. Check that voice/data/commands/ is populated.")
    test_ds = VoiceCommandDataset(test_samples, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_preds, all_labels, latencies = [], [], []

    with torch.no_grad():
        for features, label in test_loader:
            features = features.to(device)
            t0 = time.perf_counter()
            output = model(features)
            latencies.append((time.perf_counter() - t0) * 1000)

            pred = output.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label.item())

    print(f"Test set size: {len(all_labels)} clips across {len(config.LABELS)} classes\n")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(config.LABELS))))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.LABELS)))
    ax.set_yticks(range(len(config.LABELS)))
    ax.set_xticklabels(config.LABELS, rotation=45, ha="right")
    ax.set_yticklabels(config.LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (speaker-independent test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)

    report = classification_report(
        all_labels, all_preds, labels=list(range(len(config.LABELS))),
        target_names=config.LABELS, zero_division=0,
    )
    report_path = config.RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report)
    print(report)

    latencies = np.array(latencies)
    with open(config.RESULTS_DIR / "inference_latency.csv", "w") as f:
        f.write("sample_idx,latency_ms\n")
        for i, l in enumerate(latencies):
            f.write(f"{i},{l:.4f}\n")
    print(f"\nCPU inference latency: mean={latencies.mean():.2f}ms, "
          f"p95={np.percentile(latencies, 95):.2f}ms, max={latencies.max():.2f}ms")
    print(f"\nSaved: {config.RESULTS_DIR/'confusion_matrix.png'}, "
          f"{report_path}, {config.RESULTS_DIR/'inference_latency.csv'}")


if __name__ == "__main__":
    main()
