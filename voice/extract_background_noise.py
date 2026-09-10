"""
Builds the 'silence' class from REAL background noise recordings in the
full Speech Commands dataset's _background_noise_ folder, instead of
synthetic Gaussian noise. This directly targets a limitation we found in
live testing: the model trained on synthetic silence didn't reliably
recognize real room/ambient noise as silence, causing false positives.

Each output clip is a random 1-second window cut from one of the (longer)
background noise recordings (things like running tap, pink noise, white
noise, dishes, an exercise bike, a cat meowing).

Run from the project root:
    python -m voice.extract_background_noise --source_dir /path/to/extracted/speech_commands_v0.02 --num 400
"""
import argparse
import random
from pathlib import Path

import soundfile as sf

from . import config


def main():
    parser = argparse.ArgumentParser(description="Extract real background-noise silence clips.")
    parser.add_argument("--source_dir", type=str, required=True,
                         help="Path to the extracted speech_commands_v0.02 folder (contains _background_noise_/)")
    parser.add_argument("--num", type=int, default=400)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    noise_dir = Path(args.source_dir) / "_background_noise_"
    if not noise_dir.exists():
        raise FileNotFoundError(
            f"{noise_dir} not found -- make sure --source_dir points at the extracted "
            f"speech_commands_v0.02 folder (the one with 'one', 'two', etc. subfolders)."
        )
    wavs = sorted(noise_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {noise_dir}")
    print(f"Found {len(wavs)} background noise recordings: {[w.name for w in wavs]}")

    rng = random.Random(args.seed)
    out_dir = config.DATA_DIR / "silence"
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_len = config.CLIP_SAMPLES

    for i in range(args.num):
        wav_path = rng.choice(wavs)
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)  # stereo -> mono

        if sr != config.SAMPLE_RATE:
            print(f"  [warning] {wav_path.name} is {sr}Hz, expected {config.SAMPLE_RATE}Hz -- "
                  f"clip may sound off-speed. Consider resampling if this causes issues.")

        if len(data) <= clip_len:
            segment = data
        else:
            start = rng.randrange(0, len(data) - clip_len)
            segment = data[start:start + clip_len]

        sf.write(out_dir / f"real_{i}.wav", segment, sr)

    print(f"\nExtracted {args.num} real background-noise clips -> {out_dir}")


if __name__ == "__main__":
    main()
