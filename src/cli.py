"""Command-line argument parsing for the clashBot entry point."""

from __future__ import annotations

import argparse

WINDOW_TITLE = "BlueStacks App Player 1"
MODEL_ID = "troop-counter/7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="clashBot main loop")
    p.add_argument("--debug", action="store_true", help="show OpenCV debug overlay")
    p.add_argument("--record", default=None, help="JSONL path for imitation logs")
    p.add_argument("--episodes", type=int, default=1, help="matches to play")
    p.add_argument("--no-op-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--window", default=WINDOW_TITLE)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument(
        "--calibrate", action="store_true",
        help="run the interactive calibration wizard and exit",
    )
    return p.parse_args()