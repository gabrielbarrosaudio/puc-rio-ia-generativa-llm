"""
Imports the number words (one-six) from the full Google Speech Commands
v0.02 dataset into digits/data/commands/, plus a sample of other words as
'unknown'. Mirrors src/prepare_public_dataset.py but for digits instead of
directions -- see that file for the full rationale on public-dataset use,
speaker-hash tracking, and licensing (same dataset family, same license).

The 'unknown' class draws from words that are NOT one-six and NOT already
used by the direction model, so a false alarm here doesn't overlap with
words the direction model already knows to reject: 'zero', 'seven',
'eight', 'nine', 'yes', 'no'.

This script does NOT touch 'silence' -- run generate_silence.py (same
approach as the direction model: synthetic low-amplitude noise) separately.

Run from the project root:
    python -m digits.prepare_digits_dataset --source_dir /path/to/extracted/speech_commands_v0.02 --unknown_per_word 150
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from . import config

DIRECT_MAP = {word: word for word in config.NUMBERS}  # one->one, two->two, ...
# 'zero'/'seven'/'eight'/'nine'/'yes'/'no' are generic negative vocabulary.
# 'up'/'down'/'left'/'right'/'stop' are added specifically because they are
# the OTHER model's vocabulary and will genuinely be spoken while this
# model is also listening in the real game -- without them, live testing
# showed direction words like "right" being misclassified as a number.
UNKNOWN_SOURCE_WORDS = ["zero", "seven", "eight", "nine", "yes", "no",
                        "up", "down", "left", "right", "stop"]


def copy_direct_classes(source_dir: Path, manifest_rows: list):
    for src_word, dst_label in DIRECT_MAP.items():
        src_dir = source_dir / src_word
        if not src_dir.exists():
            print(f"  [skip] '{src_word}' not found in {source_dir}")
            continue
        dst_dir = config.DATA_DIR / dst_label
        dst_dir.mkdir(parents=True, exist_ok=True)

        wavs = sorted(src_dir.glob("*.wav"))
        for wav in wavs:
            dst_path = dst_dir / f"public_{wav.name}"
            if not dst_path.exists():
                shutil.copy2(wav, dst_path)
        print(f"  '{src_word}' -> digits/data/commands/{dst_label}/  ({len(wavs)} files)")
        manifest_rows.append((dst_label, "public", src_word, len(wavs)))


def copy_unknown_class(source_dir: Path, per_word: int, seed: int, manifest_rows: list):
    dst_dir = config.DATA_DIR / "unknown"
    dst_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for word in UNKNOWN_SOURCE_WORDS:
        src_dir = source_dir / word
        if not src_dir.exists():
            print(f"  [skip] '{word}' not found in {source_dir}")
            continue
        wavs = sorted(src_dir.glob("*.wav"))
        rng.shuffle(wavs)
        chosen = wavs[:per_word]
        for wav in chosen:
            dst_path = dst_dir / f"public_{word}_{wav.name}"
            if not dst_path.exists():
                shutil.copy2(wav, dst_path)
        print(f"  '{word}' -> digits/data/commands/unknown/  ({len(chosen)}/{len(wavs)} files used)")
        manifest_rows.append(("unknown", "public", word, len(chosen)))


def write_manifest(manifest_rows):
    path = config.RESULTS_DIR / "dataset_manifest.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["label", "origin", "source_word", "count"])
        writer.writerows(manifest_rows)
    print(f"\nManifest updated -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Import number words into digits/data/commands/")
    parser.add_argument("--source_dir", type=str, required=True,
                         help="Path to the extracted full speech_commands_v0.02 folder")
    parser.add_argument("--unknown_per_word", type=int, default=150,
                         help="How many clips to sample from EACH unknown-source word")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist")

    manifest_rows = []
    print("Copying number classes (one-six)...")
    copy_direct_classes(source_dir, manifest_rows)

    print(f"\nSampling {args.unknown_per_word} clips per word for 'unknown'...")
    copy_unknown_class(source_dir, args.unknown_per_word, args.seed, manifest_rows)

    write_manifest(manifest_rows)

    print("\nDone. Next: generate the silence class (no microphone needed):")
    print("    python -m digits.generate_silence --num 300")


if __name__ == "__main__":
    main()
