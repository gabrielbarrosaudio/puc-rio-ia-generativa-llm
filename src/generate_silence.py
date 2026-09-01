"""
Stage 1c - Synthetic silence generation (proof-of-concept shortcut).

The public mini_speech_commands excerpt has no "silence"/background-noise
class, and this project's scope is a proof of concept using only the public
dataset (no personal recording). This script generates synthetic "silence"
clips instead, so the model still has an explicit class to fall back on
instead of being forced to output one of the 5 command classes for every
window of audio it sees.

IMPORTANT -- document this honestly in your TCC:
Synthetic silence (Gaussian noise at varying low amplitudes) is NOT the same
as real ambient noise (fans, keyboard clicks, room echo, other people
talking, etc.). This is a reasonable simplification for a proof-of-concept
scope, but it IS a real limitation: expect more false "command detected"
triggers during real background noise than the test-set numbers alone would
suggest. State this explicitly in your Limitations / Future Work section --
it's a legitimate and expected finding, not something to hide.

Run from the project root:
    python -m src.generate_silence --num 300
"""
import argparse

import numpy as np
import soundfile as sf

from . import config


def generate_silence_clip(sr=config.SAMPLE_RATE, duration=config.CLIP_DURATION, rng=None):
    rng = rng or np.random.default_rng()
    # Vary the noise floor across clips (very quiet to slightly noisier) so
    # the class isn't a single fixed noise level the model could overfit to.
    amplitude = rng.uniform(0.0005, 0.01)
    noise = rng.normal(0, amplitude, int(sr * duration)).astype("float32")
    return np.clip(noise, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic silence clips.")
    parser.add_argument("--num", type=int, default=300,
                         help="How many synthetic silence clips to generate "
                              "(roughly matching the size of your other classes)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    out_dir = config.DATA_DIR / "silence"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.num):
        clip = generate_silence_clip(rng=rng)
        sf.write(out_dir / f"synthetic_{i}.wav", clip, config.SAMPLE_RATE)

    print(f"Generated {args.num} synthetic silence clips -> {out_dir}")
    print("These are SYNTHETIC (Gaussian noise), not real room noise.")
    print("Document this as a known limitation in your TCC's methodology/limitations section.")


if __name__ == "__main__":
    main()
