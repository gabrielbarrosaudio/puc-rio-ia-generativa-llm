"""
Board generation and movement logic for the voice-controlled board game.

Pure game logic, no Turtle/rendering here -- this is what game.py draws and
drives. Kept separate so it can be unit-tested without a display (see the
__main__ block at the bottom).

Grid convention: grid[row][col], row 0 at the top. Directions:
  up    -> row - 1
  down  -> row + 1
  left  -> col - 1
  right -> col + 1
"""
import random
from collections import deque
from enum import Enum

from . import config


class Cell(Enum):
    EMPTY = 0
    WALL = 1
    HOLE = 2


DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class Board:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def apply_move(self, pos, direction, steps):
        """Moves step-by-step in `direction`. If the next cell would be a
        wall or off the grid, the remaining steps "bounce" back in the
        opposite direction instead of being wasted (e.g. 4 free cells then
        a wall, playing a 6 -> moves 4 forward, bounces, moves 2 back).
        Bouncing can chain (bounce again if the reversed direction also
        hits something) up to a safety cap, so a token trapped between two
        walls can't loop forever.

        The goal only counts as reached if it's the FINAL resting cell
        after all steps (including any bounce) are used -- merely passing
        over it mid-move does not win. A hole, on the other hand, is fatal
        the instant it's stepped on, even mid-move.

        Returns (path, final_pos, status), where `path` is the ordered
        list of cells actually visited (for animating the real route,
        including the bounce, instead of a straight line to the end
        position) and status is one of:
          "moved"    -- used all steps, no bounce happened, didn't land on goal
          "bounced"  -- used all steps, at least one bounce happened, didn't land on goal
          "blocked"  -- got trapped bouncing and couldn't use all steps
          "hole"     -- fell into a hole -> game over/loss
          "win"      -- final resting cell is exactly the goal
        """
        dr, dc = DIRECTION_VECTORS[direction]
        row, col = pos
        path = []
        remaining = steps
        bounced = False
        bounce_attempts = 0
        max_bounce_attempts = 50
        trapped = False

        while remaining > 0:
            nrow, ncol = row + dr, col + dc
            if not self.in_bounds(nrow, ncol) or self.grid[nrow][ncol] == Cell.WALL:
                bounce_attempts += 1
                if bounce_attempts > max_bounce_attempts:
                    trapped = True
                    break
                dr, dc = -dr, -dc
                bounced = True
                continue

            row, col = nrow, ncol
            path.append((row, col))
            remaining -= 1

            if self.grid[row][col] == Cell.HOLE:
                return path, (row, col), "hole"

        if (row, col) == self.goal:
            return path, (row, col), "win"
        if trapped:
            return path, (row, col), "blocked"
        return path, (row, col), ("bounced" if bounced else "moved")

    def is_value_safe(self, pos, value):
        """A number card `value` is 'safe' from `pos` if AT LEAST ONE of
        the 4 directions, played with that value, does not end in a hole
        (the player freely chooses direction, so only one safe direction
        is needed for the value itself to count as safe)."""
        for direction in DIRECTION_VECTORS:
            _, _, status = self.apply_move(pos, direction, value)
            if status != "hole":
                return True
        return False

    def find_safe_value(self, pos, min_value, max_value):
        """Returns the first value in [min_value, max_value] that is safe
        from `pos`, or None if every value is unsafe in every direction
        (pos is completely hole-surrounded within that value range)."""
        for value in range(min_value, max_value + 1):
            if self.is_value_safe(pos, value):
                return value
        return None


def _has_path(grid, start, goal):
    """BFS over EMPTY cells only (walls AND holes are treated as
    impassable for planning purposes -- we don't want to guarantee a
    'solvable' path that requires stepping on a hole)."""
    rows, cols = len(grid), len(grid[0])
    visited = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            return True
        for dr, dc in DIRECTION_VECTORS.values():
            nrow, ncol = row + dr, col + dc
            if (0 <= nrow < rows and 0 <= ncol < cols
                    and (nrow, ncol) not in visited
                    and grid[nrow][ncol] not in (Cell.WALL, Cell.HOLE)):
                visited.add((nrow, ncol))
                queue.append((nrow, ncol))
    return False


def generate_board(rows=config.GRID_ROWS, cols=config.GRID_COLS,
                    wall_density=config.WALL_DENSITY, hole_density=config.HOLE_DENSITY,
                    min_distance=config.MIN_START_GOAL_DISTANCE,
                    seed=None, max_attempts=config.MAX_GENERATION_ATTEMPTS):
    """Generates a board with a random layout of walls/holes AND randomized
    start/goal positions (kept at least `min_distance` apart), retrying
    until a valid path from start to goal exists. Obstacles are NOT kept
    off any particular route -- the only guarantee is that *some* walkable
    path exists somewhere in the grid, so the route is genuinely not
    obvious and may wind around obstacles.
    """
    rng = random.Random(seed)

    for attempt in range(max_attempts):
        start = (rng.randrange(rows), rng.randrange(cols))
        goal = (rng.randrange(rows), rng.randrange(cols))
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < min_distance:
            continue

        grid = [[Cell.EMPTY for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                if (row, col) in (start, goal):
                    continue
                roll = rng.random()
                if roll < wall_density:
                    grid[row][col] = Cell.WALL
                elif roll < wall_density + hole_density:
                    grid[row][col] = Cell.HOLE

        if _has_path(grid, start, goal):
            return Board(grid, start, goal)

    # Backstop: guaranteed-solvable empty board with opposite corners as
    # start/goal (should be unreachable in practice given the densities
    # in config.py, but never leaves the caller without a playable board).
    start, goal = (0, 0), (rows - 1, cols - 1)
    grid = [[Cell.EMPTY for _ in range(cols)] for _ in range(rows)]
    return Board(grid, start, goal)


if __name__ == "__main__":
    # Quick sanity check, no display needed: generate many boards and
    # confirm every one is solvable, then simulate a few moves on one.
    failures = 0
    for seed in range(500):
        board = generate_board(seed=seed)
        if not _has_path(board.grid, board.start, board.goal):
            failures += 1
    print(f"Solvability check: {failures} unsolvable boards out of 500 generated")

    board = generate_board(seed=1)
    print(f"Start: {board.start}, Goal: {board.goal}")
    pos = board.start
    for direction, steps in [("right", 4), ("down", 2), ("right", 3)]:
        path, pos, status = board.apply_move(pos, direction, steps)
        print(f"  move {direction} {steps} -> path={path} final={pos} status={status}")
        if status in ("hole", "win"):
            break