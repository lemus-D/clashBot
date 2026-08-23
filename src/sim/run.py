"""CLI for the simulator: watch a match, or benchmark throughput.

    python -m src.sim.run --watch                 # visual debugger
    python -m src.sim.run --episodes 50           # headless
    python -m src.sim.run --benchmark             # steps/sec vs real play
    python -m src.sim.run --watch --no-noise      # perfect perception

In the viewer: SPACE pauses, N single-steps while paused, Q quits.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from ..env.actions import Action
from ..game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE
from .env import STEP_PERIOD_SEC, LevelSpread, ObservationNoise, SimEnv
from .opponents import OPPONENTS, make_opponent
from .units import stats_confidence_report

# Real play manages roughly this many matches an hour: ~3 min a match plus
# menu and rematch time. It is the number the sim exists to beat.
REAL_MATCHES_PER_HOUR = 18.0


class RandomSimPolicy:
    """Placeholder policy: a legal random placement, or nothing.

    Not an opponent model - the opponent pool (scripted / self-play / league)
    is deliberately still to come. This exists so the loop can be exercised
    and the viewer has something to show.
    """

    def __init__(self, no_op_prob: float = 0.7, seed: int | None = None):
        self.no_op_prob = no_op_prob
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> Action:
        if self.rng.random() < self.no_op_prob:
            return Action.no_op()
        slots = [i for i in range(HAND_SIZE) if obs["hand_playable"][i] > 0]
        if not slots:
            return Action.no_op()
        slot = self.rng.choice(slots)
        # Slot first, then the mask it implies - spells are not confined to
        # the friendly half.
        if obs["hand_is_spell"][slot] > 0:
            return Action(slot, self.rng.randrange(ARENA_COLS),
                          self.rng.randrange(ARENA_ROWS))
        tiles = np.argwhere(obs["playable_mask"] > 0)
        if len(tiles) == 0:
            return Action.no_op()
        ty, tx = tiles[self.rng.randrange(len(tiles))]
        return Action(slot, int(tx), int(ty))


def make_env(args) -> SimEnv:
    noise = ObservationNoise.off() if args.no_noise else ObservationNoise()
    return SimEnv(
        seed=args.seed,
        randomize_scale=0.0 if args.no_randomize else args.randomize,
        noise=noise,
        levels=LevelSpread.off() if args.no_levels else LevelSpread(
            troop_spread=args.level_spread, tower_spread=args.level_spread
        ),
        opponent=make_opponent(args.opponent, seed=args.seed),
    )


def run_headless(args) -> None:
    env = make_env(args)
    policy = RandomSimPolicy(seed=args.seed)
    results: dict[str, int] = {}
    total_steps = 0
    started = time.perf_counter()

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        reward_sum = 0.0
        while not done:
            obs, reward, done, info = env.step(policy(obs))
            reward_sum += reward
            total_steps += 1
        results[info["result"]] = results.get(info["result"], 0) + 1
        if args.verbose:
            print(
                f"ep {ep + 1:3d}  {info['result']:5s}  "
                f"crowns {info['crowns'][0]}-{info['crowns'][1]}  "
                f"t={info['match_time']:5.1f}s  reward={reward_sum:7.2f}  "
                f"lvl f{info['levels']['friendly']}/e{info['levels']['enemy']}"
            )

    elapsed = time.perf_counter() - started
    print(f"\n{args.episodes} episodes in {elapsed:.1f}s")
    print(f"  results        {results}")
    print(f"  steps          {total_steps} ({total_steps / elapsed:,.0f}/sec)")

    sim_seconds = total_steps * STEP_PERIOD_SEC
    speedup = sim_seconds / elapsed
    per_hour = args.episodes / elapsed * 3600
    print(f"  speed          {speedup:,.0f}x real time")
    print(
        f"  throughput     {per_hour:,.0f} matches/hour "
        f"vs ~{REAL_MATCHES_PER_HOUR:.0f} on hardware "
        f"({per_hour / REAL_MATCHES_PER_HOUR:,.0f}x)"
    )


def run_watch(args) -> None:
    import cv2

    from .render import render

    env = make_env(args)
    policy = RandomSimPolicy(seed=args.seed)
    delay = max(1, int(1000 / args.fps))
    paused = False

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            if not paused:
                obs, reward, done, info = env.step(policy(obs))
            cv2.imshow("clashBot sim", render(env))
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = not paused
            elif key == ord("n") and paused:
                obs, reward, done, info = env.step(policy(obs))
        print(
            f"ep {ep + 1}: {info['result']} "
            f"{info['crowns'][0]}-{info['crowns'][1]} at {info['match_time']:.0f}s"
        )
        # Hold the final frame so the result is readable.
        cv2.imshow("clashBot sim", render(env))
        if cv2.waitKey(1200) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="clashBot simulator")
    p.add_argument("--watch", action="store_true", help="open the visual debugger")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=float, default=12.0, help="viewer playback rate")
    p.add_argument("--no-noise", action="store_true",
                   help="perfect perception (evaluation, not training)")
    p.add_argument("--no-randomize", action="store_true",
                   help="exact stat table (evaluation, not training)")
    p.add_argument("--randomize", type=float, default=1.0,
                   help="stat randomization scale")
    p.add_argument("--benchmark", action="store_true",
                   help="throughput run: 50 episodes, headless")
    p.add_argument("--opponent", default="idle", choices=sorted(OPPONENTS),
                   help="scripted opponent to play against (default: idle, "
                        "which plays nothing - any win rate against it is "
                        "meaningless)")
    p.add_argument("--no-levels", action="store_true",
                   help="pin all card/tower levels to tournament standard "
                        "(evaluation, not training)")
    p.add_argument("--level-spread", type=int, default=1,
                   help="+/- card and tower levels sampled per episode")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--stats", action="store_true",
                   help="print how much of the unit table is actually trusted")
    args = p.parse_args()

    if args.stats:
        print(stats_confidence_report())
        return
    if args.benchmark:
        args.episodes = max(args.episodes, 50)
        run_headless(args)
        return
    if args.watch:
        run_watch(args)
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
