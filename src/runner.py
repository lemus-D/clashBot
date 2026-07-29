"""Episode loop: drives a ``ClashEnv`` with a policy until done.

Split out of ``main.py`` so the entry point stays a thin wiring layer
and the loop logic (stepping, optional debug overlay, per-episode
reporting) is reusable/testable on its own.
"""

from __future__ import annotations

import time

import cv2

from .env.environment import ClashEnv
from .policies import Policy
from .debug.overlay import render_debug_overlay


def play_episode(env: ClashEnv, policy: Policy, debug: bool = False) -> tuple[float, dict]:
    """Run one match to completion. Returns ``(total_reward, last_info)``."""
    obs = env.reset()
    done = False
    total_reward = 0.0
    info: dict = {}

    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
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

    return total_reward, info


def play_episodes(
    env: ClashEnv,
    policy: Policy,
    episodes: int,
    debug: bool = False,
    between_episodes_sec: float = 2.0,
) -> None:
    """Run ``episodes`` matches back to back, printing a summary line each."""
    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1}/{episodes} ===")
        total_reward, info = play_episode(env, policy, debug=debug)
        print(
            f"Episode complete: reward={total_reward:.2f} | "
            f"result={env.state.match_result} | "
            f"steps={info.get('step', '?')}"
        )
        time.sleep(between_episodes_sec)