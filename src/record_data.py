"""
Stage 1 - Data collection.

Records short WAV clips for each voice command class and saves them under
data/commands/<label>/<index>.wav

Run from the project root:
    python -m src.record_data --label up --num 60
    python -m src.record_data --all --num 60      # walks through every label

This is what makes your TCC results real: you are building your own labeled
dataset from your own voice. If you want to report generalization across
speakers, repeat the process with 1-2 other people and keep their clips in
the same folders (or in speaker-tagged subfolders if you want to analyze
per-speaker accuracy later).
"""
import argparse
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from . import config


def record_one_clip(duration=config.CLIP_DURATION, sr=config.SAMPLE_RATE):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def next_index(label_dir: Path) -> int:
    existing = list(label_dir.glob("*.wav"))
    if not existing:
        return 0
    indices = [int(p.stem) for p in existing if p.stem.isdigit()]
    return max(indices) + 1 if indices else 0


def record_label(label: str, num_clips: int, countdown: float = 1.0):
    label_dir = config.DATA_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_index(label_dir)

    print(f"\n=== Recording label: '{label}' ({num_clips} clips) ===")
    if label == "silence":
        print("Stay quiet / let normal background noise play during these clips.")
    elif label == "unknown":
        print("Say random words that are NOT your commands (e.g. 'hello', 'banana').")
    else:
        print(f"Say the command word: '{label}'")

    for i in range(num_clips):
        idx = start_idx + i
        print(f"[{i + 1}/{num_clips}] Recording in ", end="", flush=True)
        for t in range(int(countdown), 0, -1):
            print(t, end=" ", flush=True)
            time.sleep(1)
        print("GO!")
        audio = record_one_clip()
        out_path = label_dir / f"{idx}.wav"
        sf.write(out_path, audio, config.SAMPLE_RATE)
        print(f"  saved -> {out_path}")

    print(f"Done. '{label}' now has {len(list(label_dir.glob('*.wav')))} clips.")


def main():
    parser = argparse.ArgumentParser(description="Record voice command dataset.")
    parser.add_argument("--label", type=str, help="Single label to record (must be in config.LABELS)")
    parser.add_argument("--all", action="store_true", help="Record all labels in sequence")
    parser.add_argument("--num", type=int, default=60, help="Number of clips per label")
    parser.add_argument("--countdown", type=float, default=1.0, help="Seconds of countdown before each recording")
    args = parser.parse_args()

    if args.all:
        for label in config.LABELS:
            input(f"\nPress Enter to start recording '{label}'...")
            record_label(label, args.num, args.countdown)
    elif args.label:
        if args.label not in config.LABELS:
            raise ValueError(f"'{args.label}' not in configured labels: {config.LABELS}")
        record_label(args.label, args.num, args.countdown)
    else:
        parser.error("Provide --label <name> or --all")


if __name__ == "__main__":
    main()
