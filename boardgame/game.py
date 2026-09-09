"""
Voice-controlled board game (Mario Party-style board): get your token from
point A to point B, avoiding walls (block movement) and holes (instant
loss), using a freely-spoken direction each turn plus a number card you
must pick from your current hand of 3.

Direction: real voice recognition, reusing the trained model from
pong_voice.py (checkpoints/best_model.pt).

Number: real voice recognition too (one-six), using the digit model
trained separately in digits/ (digits/checkpoints/best_model.pt). Two
independent VoiceCommandListener instances run concurrently, each with
its own microphone stream and its own model.

Run from the project root:
    python -m boardgame.game
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import turtle

from boardgame import config
from boardgame.maze import generate_board, Cell
from boardgame.deck import NumberHand
from src.bridge import VoiceCommandListener
from digits.bridge import VoiceCommandListener as NumberVoiceListener
from digits import config as digits_config

WORD_TO_NUMBER = {word: i + 1 for i, word in enumerate(digits_config.NUMBERS)}

CELL_COLORS = {
    Cell.EMPTY: "#f4f4f4",
    Cell.WALL: "#444444",
    Cell.HOLE: "#8b0000",
}

GRID_PIXEL_W = config.GRID_COLS * config.CELL_SIZE
GRID_PIXEL_H = config.GRID_ROWS * config.CELL_SIZE
ORIGIN_X = -GRID_PIXEL_W / 2
ORIGIN_Y = GRID_PIXEL_H / 2 - 20  # leave room for the HUD text above the grid
WINDOW_WIDTH = GRID_PIXEL_W + 220
WINDOW_HEIGHT = GRID_PIXEL_H + 300


def cell_center(row, col):
    x = ORIGIN_X + col * config.CELL_SIZE + config.CELL_SIZE / 2
    y = ORIGIN_Y - row * config.CELL_SIZE - config.CELL_SIZE / 2
    return x, y


def draw_grid_lines(board, liner):
    liner.hideturtle()
    liner.speed(0)
    liner.penup()
    liner.color("#dddddd")
    liner.pensize(1)

    def dotted_line(start, end, dash_len=4, gap_len=5):
        liner.penup()
        liner.goto(start)
        liner.setheading(liner.towards(end))
        dist = liner.distance(end)
        traveled = 0
        while traveled < dist:
            liner.pendown()
            liner.forward(min(dash_len, dist - traveled))
            traveled += dash_len
            liner.penup()
            liner.forward(min(gap_len, max(0, dist - traveled)))
            traveled += gap_len

    # horizontal lines between rows
    for row in range(1, board.rows):
        y = ORIGIN_Y - row * config.CELL_SIZE
        dotted_line((ORIGIN_X, y), (ORIGIN_X + GRID_PIXEL_W, y))

    # vertical lines between columns
    for col in range(1, board.cols):
        x = ORIGIN_X + col * config.CELL_SIZE
        dotted_line((x, ORIGIN_Y), (x, ORIGIN_Y - GRID_PIXEL_H))


def draw_labels(board, labeler):
    labeler.hideturtle()
    labeler.penup()
    labeler.color("white")

    for pos, text in ((board.start, "A"), (board.goal, "B")):
        x, y = cell_center(*pos)
        labeler.goto(x, y - 12)  # nudge down so the glyph looks vertically centered
        labeler.write(text, align="center", font=("Arial", 18, "bold"))


def draw_border(board, border_turtle):
    border_turtle.hideturtle()
    border_turtle.speed(0)
    border_turtle.penup()
    border_turtle.color(CELL_COLORS[Cell.WALL])  # same color as wall cells
    border_turtle.pensize(6)

    corners = [
        (ORIGIN_X, ORIGIN_Y),
        (ORIGIN_X + GRID_PIXEL_W, ORIGIN_Y),
        (ORIGIN_X + GRID_PIXEL_W, ORIGIN_Y - GRID_PIXEL_H),
        (ORIGIN_X, ORIGIN_Y - GRID_PIXEL_H),
        (ORIGIN_X, ORIGIN_Y),
    ]
    border_turtle.goto(corners[0])
    border_turtle.pendown()
    for corner in corners[1:]:
        border_turtle.goto(corner)
    border_turtle.penup()


def ensure_safe_hand(board, pos, hand):
    """Guarantees the hand has at least one card that has a safe direction
    to play from `pos` (i.e. won't force a hole). If every card currently
    in hand would lead to a hole no matter which direction is chosen, one
    card is swapped for a value that IS safe. Does nothing if no safe
    value exists at all from this position (fully hole-surrounded) --
    that's a genuinely inescapable spot, not something a card swap fixes."""
    if any(board.is_value_safe(pos, v) for v in hand.hand):
        return
    safe_value = board.find_safe_value(pos, config.NUMBER_MIN, config.NUMBER_MAX)
    if safe_value is not None and hand.hand:
        hand.hand[0] = safe_value


def draw_board(board, stamper):
    stamper.shape("square")
    stamper.shapesize(config.CELL_SIZE / 20)
    stamper.penup()
    for row in range(board.rows):
        for col in range(board.cols):
            cell = board.grid[row][col]
            if (row, col) == board.start:
                color = "#c9a227"       # start (A): yellow
            elif (row, col) == board.goal:
                color = "#2e7d32"       # goal (B): green
            else:
                color = CELL_COLORS[cell]
            stamper.color(color)
            stamper.goto(cell_center(row, col))
            stamper.stamp()


def main():
    wn = turtle.Screen()
    wn.title("Voice Board Game (PoC)")
    wn.bgcolor("white")
    wn.setup(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    wn.tracer(0)

    board = generate_board()

    stamper = turtle.Turtle()
    stamper.hideturtle()
    stamper.speed(0)

    liner = turtle.Turtle()
    labeler = turtle.Turtle()

    def redraw_board(new_board):
        # Order matters: cell squares first, then grid lines, then labels
        # on top -- otherwise the opaque cell squares drawn for a new
        # level cover up lines/labels drawn in an earlier level.
        stamper.clearstamps()
        draw_board(new_board, stamper)
        liner.clear()
        draw_grid_lines(new_board, liner)
        labeler.clear()
        draw_labels(new_board, labeler)

    redraw_board(board)

    border = turtle.Turtle()
    draw_border(board, border)

    token = turtle.Turtle()
    token.shape("circle")
    token.color("#1565c0")
    token.penup()
    token.goto(cell_center(*board.start))

    hud = turtle.Turtle()
    hud.hideturtle()
    hud.penup()
    hud.color("black")
    hud_pos = (0, GRID_PIXEL_H / 2 + 60)

    def draw_hud(text):
        hud.clear()
        hud.goto(hud_pos)
        hud.write(text, align="center", font=("Courier", 13, "normal"))

    number_hand = NumberHand()
    ensure_safe_hand(board, board.start, number_hand)
    direction_listener = VoiceCommandListener()
    direction_listener.start()
    number_listener = NumberVoiceListener()
    number_listener.start()

    regenerate_requested = [False]

    def request_regenerate():
        regenerate_requested[0] = True

    wn.onkeypress(request_regenerate, "r")
    wn.listen()

    pos = board.start
    phase = "direction"      # "direction" -> "number" -> resolve -> back to "direction"
    pending_direction = None
    game_over = False
    level = 1
    score = 0

    def status_line():
        return (f"Fase: {level}   Pontos: {score}   Mao de numeros: {number_hand.hand}   "
                f"(restam no baralho: {number_hand.pool_left()})   [R = regenerar labirinto]")

    def prompt_line():
        if phase == "direction":
            return "Fale uma direcao: up / down / left / right"
        return f"Fale um numero da mao: {number_hand.hand}"

    draw_hud(status_line() + "\n" + prompt_line())

    try:
        while not game_over:
            wn.update()

            if regenerate_requested[0]:
                regenerate_requested[0] = False
                board = generate_board()
                pos = board.start
                ensure_safe_hand(board, pos, number_hand)
                redraw_board(board)
                token.goto(cell_center(*pos))
                phase = "direction"
                pending_direction = None
                draw_hud("Labirinto regenerado.\n" + status_line() + "\n" + prompt_line())
                continue

            voice_cmd = direction_listener.get_command()
            number_word = number_listener.get_command()
            number_cmd = WORD_TO_NUMBER.get(number_word)

            if phase == "direction" and voice_cmd in config.DIRECTIONS:
                pending_direction = voice_cmd
                phase = "number"
                draw_hud(status_line() + "\n" + prompt_line())

            elif phase == "number" and number_cmd is not None:
                if not number_hand.play(number_cmd):
                    draw_hud(status_line() + f"\nCarta {number_cmd} nao esta na mao -- tente de novo.")
                    continue

                path, new_pos, status = board.apply_move(pos, pending_direction, number_cmd)

                for cell in path:
                    token.goto(cell_center(*cell))
                    wn.update()
                    time.sleep(0.12)

                pos = new_pos
                phase = "direction"
                pending_direction = None
                ensure_safe_hand(board, pos, number_hand)

                if status == "win":
                    level += 1
                    score += 1
                    board = generate_board()
                    pos = board.start
                    number_hand = NumberHand()
                    ensure_safe_hand(board, pos, number_hand)
                    redraw_board(board)
                    token.goto(cell_center(*pos))
                    draw_hud(f"Fase {level - 1} completa! Pontos: {score}. Novo labirinto gerado.")
                    wn.update()
                    time.sleep(1.5)
                    draw_hud(status_line() + "\n" + prompt_line())
                elif status == "hole":
                    pos = board.start
                    ensure_safe_hand(board, pos, number_hand)
                    token.goto(cell_center(*pos))
                    draw_hud("Caiu em um buraco! Voltando para o inicio (ponto A).")
                    wn.update()
                    time.sleep(1.2)
                    draw_hud(status_line() + "\n" + prompt_line())
                elif number_hand.is_exhausted():
                    draw_hud(f"Baralho de numeros esgotado antes de chegar ao ponto B. Derrota. Pontos finais: {score}")
                    game_over = True
                elif status == "bounced":
                    draw_hud("Bateu na parede e quicou de volta!\n" + status_line() + "\n" + prompt_line())
                elif status == "blocked":
                    draw_hud("Ficou preso entre obstaculos, nao usou todas as casas.\n" + status_line() + "\n" + prompt_line())
                else:
                    draw_hud(status_line() + "\n" + prompt_line())

            time.sleep(0.02)

        wn.update()
        time.sleep(3)

    except KeyboardInterrupt:
        pass
    finally:
        direction_listener.stop()
        number_listener.stop()


if __name__ == "__main__":
    main()