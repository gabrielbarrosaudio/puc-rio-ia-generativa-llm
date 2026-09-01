"""
Stage 6 - Pong + voice integration.

This is your original pong.py with two additions:
  1. Import and start VoiceCommandListener (Stage 5) as a background thread.
  2. Poll listener.get_command() once per game-loop iteration and route the
     recognized command ("up"/"down"/"left"/"right"/"stop") to Paddle A.

Paddle B keeps working from the keyboard exactly as before (Up/Down arrows),
so you can always fall back to keyboard control for debugging.

NOTE: your original game only moves paddles on the Y axis. paddle_a_left /
paddle_a_right below are a minimal placeholder (they move paddle_a.setx
within screen bounds) so you can test the full voice pipeline end-to-end.
If you redesign paddle A's movement/collision logic, only these two
functions need to change -- nothing else in this file depends on them.

Run from the project root (after Stage 1-4 have produced checkpoints/best_model.pt):
    python pong_voice.py
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import turtle

from src.bridge import VoiceCommandListener

wn = turtle.Screen()
wn.title("Pong - Voice Controlled")
wn.bgcolor("pink")
wn.setup(width=800, height=600)
wn.tracer(0)

score_a = 0
score_b = 0

paddle_a = turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape("square")
paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5, stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350, 0)

paddle_b = turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5, stretch_len=1)
paddle_b.penup()
paddle_b.goto(350, 0)

ball = turtle.Turtle()
ball.speed(0)
ball.shape("square")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = 0.03
ball.dy = 0.03

pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0, 260)
pen.write("Player A: 0  Player B: 0", align="center", font=("Courier", 24, "normal"))


PADDLE_STEP = 20    # step size for a single keypress
VOICE_STEP = 60     # step size for a single voice command (bigger movement per command)


def paddle_a_up(step=PADDLE_STEP):
    paddle_a.sety(paddle_a.ycor() + step)


def paddle_a_down(step=PADDLE_STEP):
    paddle_a.sety(paddle_a.ycor() - step)


def paddle_a_left(step=PADDLE_STEP):
    # placeholder -- replace with your redesigned movement/collision logic
    paddle_a.setx(max(-380, paddle_a.xcor() - step))


def paddle_a_right(step=PADDLE_STEP):
    # placeholder -- replace with your redesigned movement/collision logic
    paddle_a.setx(min(380, paddle_a.xcor() + step))

def paddle_b_up():
    paddle_b.sety(paddle_b.ycor() + 20)


def paddle_b_down():
    paddle_b.sety(paddle_b.ycor() - 20)


VOICE_ACTIONS = {
    "up": lambda: paddle_a_up(VOICE_STEP),
    "down": lambda: paddle_a_down(VOICE_STEP),
    "left": lambda: paddle_a_left(VOICE_STEP),
    "right": lambda: paddle_a_right(VOICE_STEP),
    "stop": lambda: None,  # explicit no-op; "stop" is a real trained class, not silence
}

wn.listen()
wn.onkeypress(paddle_a_up, "w")
wn.onkeypress(paddle_a_down, "s")
wn.onkeypress(paddle_b_up, "Up")
wn.onkeypress(paddle_b_down, "Down")

listener = VoiceCommandListener()
listener.start()
print("Voice listener started. Say: up / down / left / right / stop")

try:
    while True:
        wn.update()

        command = listener.get_command()
        if command in VOICE_ACTIONS:
            print(f"[voice] {command}")
            VOICE_ACTIONS[command]()

        ball.setx(ball.xcor() + ball.dx)
        ball.sety(ball.ycor() + ball.dy)

        if ball.ycor() > 290:
            ball.sety(290)
            ball.dy *= -1
        if ball.ycor() < -290:
            ball.sety(-290)
            ball.dy *= -1

        if ball.xcor() > 390:
            ball.goto(0, 0)
            ball.dx *= -1
            score_a += 1
            paddle_a.goto(-350, 0)
            paddle_b.goto(350, 0)
            pen.clear()
            pen.write(f"Player A: {score_a}  Player B: {score_b}", align="center", font=("Courier", 24, "normal"))
            ball.dx = 0.03

        if ball.xcor() < -390:
            ball.goto(0, 0)
            ball.dx *= -1
            score_b += 1
            paddle_a.goto(-350, 0)
            paddle_b.goto(350, 0)
            pen.clear()
            pen.write(f"Player A: {score_a}  Player B: {score_b}", align="center", font=("Courier", 24, "normal"))
            ball.dx = 0.03

        if (ball.xcor() > 340 and ball.xcor() < 350) and (ball.ycor() < paddle_b.ycor() + 50 and ball.ycor() > paddle_b.ycor() - 50):
            ball.setx(340)
            ball.dx *= -1.2

        if (ball.xcor() < -340 and ball.xcor() > -350) and (ball.ycor() < paddle_a.ycor() + 50 and ball.ycor() > paddle_a.ycor() - 50):
            ball.setx(-340)
            ball.dx *= -1.2

except KeyboardInterrupt:
    pass
finally:
    listener.stop()
