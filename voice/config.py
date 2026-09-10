"""
Configuration for the UNIFIED voice command model -- replaces the two
separate models in src/ (direction) and digits/ (number). Those two
specialist models each had a weak, thin 'unknown' class that never truly
learned the other model's vocabulary, causing real confusion in live
testing (e.g. 'left' misclassified as 'one'). This model instead learns
all the real vocabulary as genuine, distinct, well-represented classes.

src/ and digits/ are left in place (not deleted) as a record of the
earlier approach and its documented limitations -- useful material for
the TCC's methodology/discussion section. The game now uses voice/ only.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "voice" / "data" / "commands"
RESULTS_DIR = PROJECT_ROOT / "voice" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "voice" / "checkpoints"

for d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Audio ---
SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# --- Labels ---
DIRECTIONS = ["up", "down", "left", "right", "stop"]
NUMBERS = ["one", "two", "three", "four", "five", "six"]
LABELS = DIRECTIONS + NUMBERS + ["silence", "unknown"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

# --- Features ---
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# --- Real-time inference ---
CONFIDENCE_THRESHOLD = 0.85
ENERGY_THRESHOLD = 0.02
COOLDOWN_SECONDS = 0.6
