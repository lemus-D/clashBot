"""Entry point: drive ``ClashEnv`` with a pluggable policy.

The default policy is ``RandomPolicy`` which picks any (affordable,
placeable) action. Swap in your trained policy by replacing the
``policy`` callable below; the contract is ``policy(obs) -> action``
where ``action`` is either an ``Action`` instance, an integer index
into the discrete action space, or a ``(hand, x, y)`` tuple.

Usage::

    python -m src.main
    python -m src.main --debug
    python -m src.main --record logs/run.jsonl
    python -m src.main --episodes 5 --record logs/run.jsonl
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Callable, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

from .actions import Action, action_space_size
from .environment import ClashEnv
from .gameBoard import HAND_SIZE, ARENA_COLS, ARENA_ROWS
from .windowCap import render_debug_overlay


WINDOW_TITLE = "BlueStacks App Player 1"
MODEL_ID = "troop-counter/7"


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="clashBot main loop")
    p.add_argument("--debug", action="store_true", help="show OpenCV debug overlay")
    p.add_argument("--record", default=None, help="JSONL path for imitation logs")
    p.add_argument("--episodes", type=int, default=1, help="matches to play")
    p.add_argument("--no-op-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--window", default=WINDOW_TITLE)
    p.add_argument("--model", default=MODEL_ID)
    return p.parse_args()


def run() -> None:
    args = parse_args()
    load_dotenv()

    policy: Policy = RandomPolicy(no_op_prob=args.no_op_prob, seed=args.seed)

    env = ClashEnv(
        window_title=args.window,
        model_id=args.model,
        record_path=args.record,
    )

    print(f"Action space size: {action_space_size()}")
    print(
        f"Arena grid: {ARENA_ROWS} rows x {ARENA_COLS} cols, "
        f"hand size: {HAND_SIZE}"
    )

    try:
        for episode in range(args.episodes):
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
            obs = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward

                if args.debug and env._frame is not None and env.board is not None:
                    overlay = render_debug_overlay(
                        env._frame,
                        env.board,
                        env.state,
                        lifecycle_state=info.get("lifecycle_state"),
                    )
                    cv2.imshow("clashBot debug", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        done = True

            print(
                f"Episode complete: reward={total_reward:.2f} | "
                f"result={env.state.match_result} | "
                f"steps={info.get('step', '?')}"
            )

            time.sleep(2.0)
    finally:
        if args.debug:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    run()
