"""
Stage 1b - Integrating an external public dataset (e.g. mini_speech_commands).

This script copies wav files from an extracted public dataset into this
project's data/commands/<label>/ structure, so the rest of the pipeline
(dataset.py, train.py, evaluate.py) works unchanged regardless of where the
audio originally came from.

Design choices, and why:
  - Files copied from the public dataset are prefixed "public_" (e.g.
    public_988e2f9a_nohash_0.wav). Files you record yourself via
    record_data.py are plain numeric names (0.wav, 1.wav, ...). This lets
    you tell the two sources apart later just from the filename -- see
    dataset.sample_origin() -- which is what makes the "public vs. personal"
    domain-gap comparison in evaluate.py possible.
  - "go", "no", "yes" are mapped to the "unknown" class. You control how
    many clips per word are sampled with --unknown_per_word, so the
    "unknown" class doesn't end up 3x larger than the real command classes.
  - This script does NOT touch "silence" -- generate it separately with
    generate_silence.py (synthetic, no microphone needed):
        python -m src.generate_silence --num 300

Run from the project root (after extracting the dataset zip somewhere):
    python -m src.prepare_public_dataset \
        --source_dir /path/to/extracted/mini_speech_commands \
        --unknown_per_word 400
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from . import config

# public dataset folder name -> project label
DIRECT_MAP = {"up": "up", "down": "down", "left": "left", "right": "right", "stop": "stop"}
UNKNOWN_SOURCE_WORDS = ["go", "no", "yes"]


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
        print(f"  '{src_word}' -> data/commands/{dst_label}/  ({len(wavs)} files)")
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
        print(f"  '{word}' -> data/commands/unknown/  ({len(chosen)}/{len(wavs)} files used)")
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
    print("(Include this table in your TCC methodology section: it documents")
    print(" exactly how many samples per class came from which source.)")


def main():
    parser = argparse.ArgumentParser(description="Import an external command dataset into data/commands/")
    parser.add_argument("--source_dir", type=str, required=True,
                         help="Path to the extracted public dataset folder (contains up/, down/, go/, ...)")
    parser.add_argument("--unknown_per_word", type=int, default=400,
                         help="How many clips to sample from EACH of go/no/yes for the 'unknown' class")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist")

    manifest_rows = []
    print("Copying direct command classes (up/down/left/right/stop)...")
    copy_direct_classes(source_dir, manifest_rows)

    print(f"\nSampling {args.unknown_per_word} clips per word for 'unknown' (go/no/yes)...")
    copy_unknown_class(source_dir, args.unknown_per_word, args.seed, manifest_rows)

    write_manifest(manifest_rows)

    print("\nDone. Next: generate the silence class (no microphone needed):")
    print("    python -m src.generate_silence --num 300")


if __name__ == "__main__":
    main()
