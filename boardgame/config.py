"""
Configuration for the voice-controlled board game (separate from the Pong
project -- this is a second, independent game in the same repo).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Grid ---
GRID_COLS = 8
GRID_ROWS = 6
CELL_SIZE = 60  # pixels

# --- Obstacle generation ---
WALL_DENSITY = 0.15   # fraction of non-start/goal cells that become walls
HOLE_DENSITY = 0.10   # fraction of non-start/goal cells that become holes
MAX_GENERATION_ATTEMPTS = 300  # regenerate until a valid A->B path exists
MIN_START_GOAL_DISTANCE = 6    # manhattan distance, keeps A/B from spawning too close

# --- Cards ---
# Direction is chosen freely by voice each turn -- no deck/limit.
# Number is the strategic resource: a hand of NUMBER_HAND_SIZE cards drawn
# from a pool of NUMBER_DECK_SIZE total cards for the whole game.
NUMBER_HAND_SIZE = 3
NUMBER_DECK_SIZE = 20
NUMBER_MIN = 1
NUMBER_MAX = 6

# --- Voice ---
DIRECTIONS = ["up", "down", "left", "right"]