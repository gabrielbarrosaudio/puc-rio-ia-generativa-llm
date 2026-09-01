"""
Stage 2 - PyTorch Dataset for the voice command clips, plus stratified
train/val/test splitting and light data augmentation for the training split.
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import config
from .features import load_and_fix_length, waveform_to_logmel


class VoiceCommandDataset(Dataset):
    def __init__(self, samples, augment=False):
        """
        samples: list of (filepath, label_idx) tuples
        augment: if True, applies random noise + time-shift.
                 Only set True for the TRAINING split -- val/test must stay
                 untouched so your reported metrics reflect real performance.
        """
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # Random time shift (+-100ms) -- makes the model robust to imprecise
        # start-of-word timing, which matters a lot for real-time streaming.
        shift = int(random.uniform(-0.1, 0.1) * config.SAMPLE_RATE)
        waveform = torch.roll(waveform, shifts=shift, dims=1)

        # Random Gaussian noise -- simulates mic/room noise variability.
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


def sample_origin(path: Path) -> str:
    """Returns 'public' for clips imported via prepare_public_dataset.py
    (filenames prefixed 'public_'), or 'personal' for clips recorded via
    record_data.py (plain numeric filenames like 0.wav, 1.wav, ...).

    Useful for reporting a train/test domain-gap in your TCC: e.g. "the
    model reaches X% accuracy on held-out public clips but only Y% on
    clips recorded with my own microphone."
    """
    return "public" if Path(path).name.startswith("public_") else "personal"


def extract_speaker_id(path) -> str:
    """Extracts the speaker hash from a mini_speech_commands-style filename
    (e.g. '988e2f9a_nohash_0.wav' -> '988e2f9a'), stripping the 'public_'
    and, for the unknown class, the source-word prefixes added by
    prepare_public_dataset.py. Falls back to the filename itself for clips
    that don't follow this naming (personal recordings, synthetic silence),
    which is fine: without a real speaker ID each such file is simply
    treated as its own group, which cannot cause train/test leakage.
    """
    name = Path(path).stem
    if name.startswith("public_"):
        name = name[len("public_"):]
    for prefix in ("go_", "no_", "yes_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if "_nohash_" in name:
        return name.split("_nohash_")[0]
    return name


def split_dataset_by_speaker(samples, val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT, seed=config.RANDOM_SEED):
    """Speaker-independent split: every speaker ID is assigned to exactly
    ONE of train/val/test, so no voice heard during training is ever
    evaluated on. This is the split to use whenever you want to claim the
    model generalizes to unseen speakers (as opposed to split_dataset(),
    which only guarantees balanced classes and CAN leak the same speaker
    across splits).
    """
    rng = random.Random(seed)
    speakers = sorted({extract_speaker_id(path) for path, _ in samples})
    rng.shuffle(speakers)

    n = len(speakers)
    n_val = max(1, int(n * val_split))
    n_test = max(1, int(n * test_split))
    val_speakers = set(speakers[:n_val])
    test_speakers = set(speakers[n_val:n_val + n_test])
    # everyone else -> train

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


def scan_dataset(data_dir: Path = config.DATA_DIR):
    """Walks data/commands/<label>/*.wav and returns (path, label_idx) tuples."""
    samples = []
    for label in config.LABELS:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        for wav_path in sorted(label_dir.glob("*.wav")):
            samples.append((wav_path, config.LABEL_TO_IDX[label]))
    return samples


def split_dataset(samples, val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT, seed=config.RANDOM_SEED):
    """Stratified split: each label is split independently so every class is
    represented proportionally in train/val/test, even with a small dataset."""
    rng = random.Random(seed)
    by_label = {}
    for path, label_idx in samples:
        by_label.setdefault(label_idx, []).append((path, label_idx))

    train, val, test = [], [], []
    for label_idx, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(n * val_split)) if n > 3 else 0
        n_test = max(1, int(n * test_split)) if n > 3 else 0
        val.extend(items[:n_val])
        test.extend(items[n_val:n_val + n_test])
        train.extend(items[n_val + n_test:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
