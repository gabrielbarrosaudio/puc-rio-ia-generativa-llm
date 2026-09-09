"""
Configuration for the NUMBER-card voice model. Deliberately a separate
package from src/ (the up/down/left/right/stop model): different label
set, different training run, different checkpoint -- keeping it isolated
means nothing here can break the direction model that's already working.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "digits" / "data" / "commands"
RESULTS_DIR = PROJECT_ROOT / "digits" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "digits" / "checkpoints"

for d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Audio (same conventions as src/config.py, so the two models are
# directly comparable in the TCC write-up) ---
SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# --- Labels ---
NUMBERS = ["one", "two", "three", "four", "five", "six"]
LABELS = NUMBERS + ["silence", "unknown"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

# --- Features (identical to the direction model) ---
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
CONFIDENCE_THRESHOLD = 0.75
COOLDOWN_SECONDS = 0.4
