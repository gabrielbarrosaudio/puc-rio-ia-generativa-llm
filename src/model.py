"""
Stage 3 - Model definition.

A small CNN over log-Mel spectrograms. This architecture (a few conv+pool
blocks, then global average pooling and a linear classifier) is standard
for keyword spotting: it is cheap enough to run in real time on a laptop
CPU, which Stage 5/6 requires.

For your TCC, this is a good baseline to justify empirically -- e.g. you
could compare it against a deeper CNN or a CNN+GRU variant and report
accuracy vs. inference-latency trade-offs using the scripts in evaluate.py.
"""
import torch.nn as nn

from . import config


class VoiceCommandCNN(nn.Module):
    def __init__(self, n_classes=len(config.LABELS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pool -> robust to variable time length
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 1, n_mels, time_frames)
        x = self.features(x)
        x = self.classifier(x)
        return x
