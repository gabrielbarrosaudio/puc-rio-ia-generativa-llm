"""
Imports ALL the direct vocabulary (up/down/left/right/stop AND one-six)
from the full Google Speech Commands v0.02 dataset as genuine classes,
plus a sample of the REMAINING words as 'unknown'. This is the key
difference from the old two-model setup: every word that can realistically
be spoken during the game is either a real class or explicitly represented
in 'unknown' -- nothing is vocabulary the model has simply never seen.

'unknown' draws from: yes, no, on, off, go, zero, seven, eight, nine, and
the auxiliary words (bed, bird, cat, dog, happy, house, marvin, sheila,
tree, wow) -- 19 different words, much more diverse than before, which
should also give the model a better real decision boundary for 'unknown'
in general (more views of "what a non-command word sounds like").

This script does NOT touch 'silence' -- run extract_background_noise.py
separately (real background noise this time, not synthetic).

Run from the project root:
    python -m voice.prepare_dataset --source_dir /path/to/extracted/speech_commands_v0.02 --unknown_per_word 220
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from . import config

DIRECT_WORDS = config.DIRECTIONS + config.NUMBERS
UNKNOWN_SOURCE_WORDS = [
    "yes", "no", "on", "off", "go", "zero", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
]


def copy_direct_classes(source_dir: Path, manifest_rows: list):
    for word in DIRECT_WORDS:
        src_dir = source_dir / word
        if not src_dir.exists():
            print(f"  [skip] '{word}' not found in {source_dir}")
            continue
        dst_dir = config.DATA_DIR / word
        dst_dir.mkdir(parents=True, exist_ok=True)

        wavs = sorted(src_dir.glob("*.wav"))
        for wav in wavs:
            dst_path = dst_dir / f"public_{wav.name}"
            if not dst_path.exists():
                shutil.copy2(wav, dst_path)
        print(f"  '{word}' -> voice/data/commands/{word}/  ({len(wavs)} files)")
        manifest_rows.append((word, "public", word, len(wavs)))


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
        print(f"  '{word}' -> voice/data/commands/unknown/  ({len(chosen)}/{len(wavs)} files used)")
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
    parser = argparse.ArgumentParser(description="Import the unified voice vocabulary.")
    parser.add_argument("--source_dir", type=str, required=True,
                         help="Path to the extracted full speech_commands_v0.02 folder")
    parser.add_argument("--unknown_per_word", type=int, default=220,
                         help="Clips sampled per unknown-source word (19 words -> ~4000+ total, "
                              "comparable in size to each real class)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist")

    manifest_rows = []
    print("Copying direct classes (up/down/left/right/stop/one-six)...")
    copy_direct_classes(source_dir, manifest_rows)

    print(f"\nSampling {args.unknown_per_word} clips per word for 'unknown' ({len(UNKNOWN_SOURCE_WORDS)} words)...")
    copy_unknown_class(source_dir, args.unknown_per_word, args.seed, manifest_rows)

    write_manifest(manifest_rows)

    print("\nDone. Next: extract real background noise for 'silence':")
    print("    python -m voice.extract_background_noise --source_dir <same path> --num 400")


if __name__ == "__main__":
    main()
