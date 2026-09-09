"""
Placeholder for number-card voice input.

There is no trained digit-recognition model yet (the mini_speech_commands
excerpt used for direction has no digit words -- the full Google Speech
Commands v0.02 dataset does, see the note in README.md). Until that model
exists, number cards are played with the keyboard (number keys 1-6)
instead of voice.

Deliberately mirrors VoiceCommandListener's non-blocking get_command()
interface (src/bridge.py) so swapping this out for real voice recognition
later is a one-line change in game.py -- nothing else needs to change.
"""


class NumberInputPlaceholder:
    def __init__(self, screen, min_value=1, max_value=6):
        self._pending = None
        for digit in range(min_value, max_value + 1):
            screen.onkeypress(self._make_handler(digit), str(digit))
        screen.listen()

    def _make_handler(self, digit):
        def handler():
            self._pending = digit
        return handler

    def get_command(self):
        """Non-blocking. Returns the next pressed digit (int), or None."""
        value = self._pending
        self._pending = None
        return value
