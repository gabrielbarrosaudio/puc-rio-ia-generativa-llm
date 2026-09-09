"""
Synthetic silence generation for the digit model (mirrors
src/generate_silence.py -- same rationale: no microphone needed, but
document the synthetic-vs-real-noise limitation in the TCC).

Run from the project root:
    python -m digits.generate_silence --num 300
"""
import argparse

import numpy as np
import soundfile as sf

from . import config


def generate_silence_clip(sr=config.SAMPLE_RATE, duration=config.CLIP_DURATION, rng=None):
    rng = rng or np.random.default_rng()
    amplitude = rng.uniform(0.0005, 0.01)
    noise = rng.normal(0, amplitude, int(sr * duration)).astype("float32")
    return np.clip(noise, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic silence clips for the digit model.")
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    out_dir = config.DATA_DIR / "silence"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.num):
        clip = generate_silence_clip(rng=rng)
        sf.write(out_dir / f"synthetic_{i}.wav", clip, config.SAMPLE_RATE)

    print(f"Generated {args.num} synthetic silence clips -> {out_dir}")


if __name__ == "__main__":
    main()