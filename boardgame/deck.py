"""
Card system for the voice-controlled board game.

Direction: chosen freely by voice each turn -- NOT a deck, no limit.

Number: the strategic resource. The player always holds a hand of
NUMBER_HAND_SIZE cards (values visible, e.g. "2, 5, 6"), and must speak a
number that's actually IN their current hand to play it -- this is what
makes it a real choice ("pick whichever of my 3 cards gets me closest to
B without hitting a wall/hole") rather than free choice, which is the
whole point per your description. Playing a card removes it from hand and
immediately draws a replacement from the remaining pool (if any left).
When the pool AND hand are both exhausted, no more moves are possible; if
the goal hasn't been reached, that's a loss.
"""
import random

from . import config


class NumberHand:
    def __init__(self, pool_size=config.NUMBER_DECK_SIZE, hand_size=config.NUMBER_HAND_SIZE,
                 min_value=config.NUMBER_MIN, max_value=config.NUMBER_MAX, seed=None):
        rng = random.Random(seed)
        values = list(range(min_value, max_value + 1))
        # total pool of cards for the whole game, already "shuffled"
        self._remaining = [rng.choice(values) for _ in range(pool_size)]
        self.hand = []
        for _ in range(hand_size):
            self._draw_into_hand()

    def _draw_into_hand(self):
        if self._remaining:
            self.hand.append(self._remaining.pop())

    def pool_left(self) -> int:
        return len(self._remaining)

    def is_exhausted(self) -> bool:
        """True when there is nothing left to play at all."""
        return len(self.hand) == 0

    def play(self, value) -> bool:
        """Tries to play `value` from the current hand. Returns True and
        consumes+replenishes on success; returns False (hand unchanged)
        if `value` isn't actually in hand right now -- that's an invalid
        move, not a crash, so the caller can ask the player to try again."""
        if value not in self.hand:
            return False
        self.hand.remove(value)
        self._draw_into_hand()
        return True


if __name__ == "__main__":
    hand = NumberHand(pool_size=15, hand_size=3, seed=1)
    print(f"Initial hand: {hand.hand}, pool left: {hand.pool_left()}")

    turn = 0
    while not hand.is_exhausted():
        turn += 1
        choice = hand.hand[0]  # simulate: always play the first card in hand
        ok = hand.play(choice)
        print(f"  turn {turn}: played {choice} ({'ok' if ok else 'invalid'}) "
              f"-> hand now {hand.hand}, pool left: {hand.pool_left()}")

    # demonstrate an invalid play
    hand2 = NumberHand(pool_size=5, hand_size=3, seed=2)
    print(f"\nHand: {hand2.hand}")
    bad_value = next(v for v in range(1, 7) if v not in hand2.hand)
    print(f"Trying to play {bad_value} (not in hand): {hand2.play(bad_value)}")
