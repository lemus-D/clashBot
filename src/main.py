"""Entry point: drive ``ClashEnv`` with a pluggable policy.

The default policy is ``RandomPolicy`` (see ``policies.py``), which
picks any (affordable, placeable) action. Swap in your trained policy
by constructing something else with the ``policy(obs) -> action``
contract and passing it to ``play_episodes`` below.

Usage::

    python -m src.main
    python -m src.main --debug
    python -m src.main --record logs/run.jsonl
    python -m src.main --episodes 5 --record logs/run.jsonl
    python -m src.main --calibrate
"""

from __future__ import annotations

import cv2
from dotenv import load_dotenv

from .cli import parse_args
from .env.actions import action_space_size
from .env.environment import ClashEnv
from .game.board import HAND_SIZE, ARENA_COLS, ARENA_ROWS
from .policies import RandomPolicy
from .runner import play_episodes


def run() -> None:
    args = parse_args()
    load_dotenv()

    if args.calibrate:
        from .calibrate import calibrate
        calibrate(args.window)
        return

    policy = RandomPolicy(no_op_prob=args.no_op_prob, seed=args.seed)
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
        play_episodes(env, policy, args.episodes, debug=args.debug)
    finally:
        if args.debug:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    run()