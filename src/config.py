"""
Central configuration shared across all scripts in the voice-pong project.
Keeping these values in one place avoids mismatches between training and
real-time inference (e.g. training with 16kHz audio but recording at 44.1kHz).
"""
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "commands"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Audio ---
SAMPLE_RATE = 16000          # Hz. 16kHz is standard for speech models and keeps files small.
CLIP_DURATION = 1.0          # seconds per recorded/inferred command
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# --- Labels ---
# "silence" and "unknown" are NOT optional: without them the model has no way
# to say "no command was spoken", and it will hallucinate a class during
# background noise, breathing, keyboard clicks, or the game's own sounds.
COMMANDS = ["up", "down", "left", "right", "stop"]
LABELS = COMMANDS + ["silence", "unknown"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

# --- Features ---
N_MELS = 40
N_FFT = 400          # 25ms window at 16kHz
HOP_LENGTH = 160     # 10ms hop at 16kHz

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# --- Real-time inference ---
CONFIDENCE_THRESHOLD = 0.75   # minimum softmax probability to accept a command
COOLDOWN_SECONDS = 1.0        # minimum time between accepted commands (debounce)
