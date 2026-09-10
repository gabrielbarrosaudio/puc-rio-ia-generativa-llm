"""
Stage 2 - PyTorch Dataset for the unified voice model, plus speaker-
independent train/val/test splitting and augmentation.
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import config
from .features import load_and_fix_length, waveform_to_logmel


class VoiceCommandDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        shift = int(random.uniform(-0.1, 0.1) * config.SAMPLE_RATE)
        waveform = torch.roll(waveform, shifts=shift, dims=1)
        if random.random() < 0.5:
            noise_level = random.uniform(0.0, 0.01)
            waveform = waveform + noise_level * torch.randn_like(waveform)
        return waveform

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        waveform = load_and_fix_length(path)
        if self.augment:
            waveform = self._augment(waveform)
        features = waveform_to_logmel(waveform)
        return features, label_idx


def scan_dataset(data_dir: Path = config.DATA_DIR):
    samples = []
    for label in config.LABELS:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        for wav_path in sorted(label_dir.glob("*.wav")):
            samples.append((wav_path, config.LABEL_TO_IDX[label]))
    return samples


def extract_speaker_id(path) -> str:
    """Handles both 'public_<hash>_nohash_<n>.wav' (direct classes) and
    'public_<sourceword>_<hash>_nohash_<n>.wav' (unknown class, prefixed
    with whichever of the 19 source words it came from) without needing
    to enumerate those words here -- whatever sits right before
    '_nohash_' is treated as the hash if there's no further underscore,
    or the last underscore-separated token if there is one."""
    name = Path(path).stem
    if name.startswith("public_"):
        name = name[len("public_"):]
    if "_nohash_" in name:
        prefix = name.split("_nohash_")[0]
        return prefix.split("_")[-1]
    return name


def split_dataset_by_speaker(samples, val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT, seed=config.RANDOM_SEED):
    rng = random.Random(seed)
    speakers = sorted({extract_speaker_id(path) for path, _ in samples})
    rng.shuffle(speakers)

    n = len(speakers)
    n_val = max(1, int(n * val_split))
    n_test = max(1, int(n * test_split))
    val_speakers = set(speakers[:n_val])
    test_speakers = set(speakers[n_val:n_val + n_test])

    train, val, test = [], [], []
    for path, label_idx in samples:
        spk = extract_speaker_id(path)
        if spk in val_speakers:
            val.append((path, label_idx))
        elif spk in test_speakers:
            test.append((path, label_idx))
        else:
            train.append((path, label_idx))

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
