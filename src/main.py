"""Entry point: drive ``ClashEnv`` with a policy (``policy(obs) -> Action``).

Usage::

    python -m src.main
    python -m src.main --debug
    python -m src.main --episodes 5 --record logs/run.jsonl
    python -m src.main --record-human demos/run.jsonl --episodes 5
    python -m src.main --policy imitation --weights models/imitation.pt
    python -m src.main --calibrate
    python -m src.main --calibrate timer
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from dotenv import load_dotenv

from .calibrate import ALL_PHASES, PHASE_NAMES, resolve_phases
from .env.actions import Action, action_space_size
from .env.environment import ClashEnv
from .game.board import HAND_SIZE, ARENA_COLS, ARENA_ROWS
from .debug.overlay import render_debug_overlay


WINDOW_TITLE = "BlueStacks App Player 1"
MODEL_ID = "troop-counter/8"
DEBUG_RECORD_DIR = "logs"


class RandomPolicy:
    """Picks a uniformly random valid action, or NO_OP if none exist.

    Validity = card slot non-empty, elixir >= cost, tile in playable
    mask. Same checks ``ActionExecutor`` runs - this just avoids wasted
    drag attempts.
    """

    def __init__(self, no_op_prob: float = 0.5, seed: int | None = None):
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
    p.add_argument(
        "--debug", action="store_true",
        help=f"show OpenCV debug overlay; also records to {DEBUG_RECORD_DIR}/ "
             f"unless --record gives an explicit path",
    )
    p.add_argument("--record", default=None, help="JSONL path for imitation logs")
    p.add_argument(
        "--record-human", default=None, metavar="PATH",
        help="record human play to this JSONL instead of running a policy",
    )
    p.add_argument(
        "--policy", choices=("random", "imitation"), default="random",
    )
    p.add_argument(
        "--weights", default=None, metavar="PATH",
        help="checkpoint for --policy imitation",
    )
    p.add_argument("--episodes", type=int, default=1, help="matches to play")
    p.add_argument("--no-op-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--window", default=WINDOW_TITLE)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument(
        "--calibrate", nargs="?", const=ALL_PHASES, default=None, metavar="PHASE",
        help="run the interactive calibration wizard and exit; with no value "
             f"runs every phase, or name one of: {', '.join(PHASE_NAMES)}",
    )
    args = p.parse_args()
    if args.policy == "imitation" and not args.weights:
        p.error("--policy imitation requires --weights")
    if args.calibrate is not None:
        try:
            resolve_phases(args.calibrate)
        except ValueError as exc:
            p.error(str(exc))
    return args


def resolve_record_path(record: str | None, debug: bool) -> str | None:
    """Decide where this run records, creating the parent directory.

    An explicit ``--record`` always wins. ``--debug`` on its own records to
    a timestamped default so a misbehaving debug run can be reviewed after
    the fact and consecutive runs never overwrite each other. Returns None
    when neither flag asks for a recording.
    """
    if not record:
        if not debug:
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        record = str(Path(DEBUG_RECORD_DIR) / f"debug-{stamp}.jsonl")
    Path(record).parent.mkdir(parents=True, exist_ok=True)
    return record


def run_episode(env: ClashEnv, policy: Callable[[dict], Action], debug: bool) -> None:
    """Play one episode to completion and print its summary."""
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        obs, reward, done, info = env.step(policy(obs))
        total_reward += reward

        if debug and env._frame is not None and env.board is not None:
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


def run() -> None:
    args = parse_args()
    load_dotenv()

    if args.calibrate is not None:
        from .calibrate import calibrate
        calibrate(args.window, args.calibrate)
        return

    if args.record_human:
        from .imitation.recorder import record_demos
        record_demos(
            window_title=args.window,
            model_id=args.model,
            record_path=args.record_human,
            episodes=args.episodes,
        )
        return

    if args.policy == "imitation":
        from .imitation.policy import ImitationPolicy
        policy: Callable[[dict], Action] = ImitationPolicy(args.weights)
    else:
        policy = RandomPolicy(no_op_prob=args.no_op_prob, seed=args.seed)
    record_path = resolve_record_path(args.record, args.debug)
    if record_path:
        print(f"Recording to: {record_path}")
    env = ClashEnv(
        window_title=args.window,
        model_id=args.model,
        record_path=record_path,
    )

    print(f"Action space size: {action_space_size()}")
    print(
        f"Arena grid: {ARENA_ROWS} rows x {ARENA_COLS} cols, "
        f"hand size: {HAND_SIZE}"
    )

    try:
        for episode in range(args.episodes):
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")
            run_episode(env, policy, debug=args.debug)
            time.sleep(2.0)
    finally:
        if args.debug:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    run()
