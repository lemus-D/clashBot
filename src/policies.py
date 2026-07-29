"""Pluggable policy interface, plus a random baseline policy.

A policy is any callable ``policy(obs) -> action``, where ``action``
is either an ``Action`` instance, an integer index into the discrete
action space, or a ``(hand, x, y)`` tuple. Swap ``RandomPolicy`` out
for your trained policy wherever ``main.run()`` constructs one.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np

from .env.actions import Action
from .game.board import HAND_SIZE

Policy = Callable[[dict], object]


class RandomPolicy:
    """Picks a uniformly random valid action, or NO_OP if none exist.

    Validity = card slot non-empty, elixir >= cost, tile in playable
    mask. Same checks ``ActionExecutor`` runs - this just avoids wasted
    drag attempts.
    """

    def __init__(self, no_op_prob: float = 0.5, seed: Optional[int] = None):
        self.no_op_prob = no_op_prob
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> Action:
        if self.rng.random() < self.no_op_prob:
            return Action.no_op()

        playable = obs["hand_playable"]
        valid_slots = [i for i in range(HAND_SIZE) if playable[i] > 0]
        if not valid_slots:
            return Action.no_op()

        mask = obs["playable_mask"]
        valid_tiles = np.argwhere(mask > 0)
        if len(valid_tiles) == 0:
            return Action.no_op()

        slot = self.rng.choice(valid_slots)
        ty, tx = valid_tiles[self.rng.randrange(len(valid_tiles))]
        return Action(hand_index=int(slot), tile_x=int(tx), tile_y=int(ty))