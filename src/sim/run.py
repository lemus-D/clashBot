"""CLI for the simulator: watch a match, or benchmark throughput.

    python -m src.sim.run --watch                       # visual debugger
    python -m src.sim.run --watch --policy models/ppo.pt --opponent all
    python -m src.sim.run --episodes 50                 # headless
    python -m src.sim.run --benchmark                   # steps/sec vs real play
    python -m src.sim.run --watch --no-noise            # perfect perception

In the viewer: SPACE pauses, N single-steps while paused, Q quits.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from ..env.actions import Action
from ..game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE
from .env import (
    ArchetypeDeckSpread,
    DeckSpread,
    STEP_PERIOD_SEC,
    LevelSpread,
    ObservationNoise,
    SimEnv,
)
from .opponents import BASELINE_POOL, OPPONENTS, PUNISHER_POOL, make_opponent
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


class CheckpointPolicy:
    """A trained RL checkpoint, wrapped as the viewer's obs -> Action callable.

    DETERMINISTIC by default, because that is what ``rl.evaluate`` measures -
    the win rates a checkpoint is quoted at are argmax play. Watching the
    sampled policy is a different, noisier thing, so it is opt-in.

    Torch is imported lazily: the headless simulator and the benchmark must
    stay runnable without it.
    """

    def __init__(self, path: str, deterministic: bool = True):
        import torch

        from ..env.observation import ObservationBuilder
        from ..rl.policy import to_action
        from ..rl.train import load_checkpoint

        self._torch = torch
        self._to_action = to_action
        self._flatten = ObservationBuilder.flatten
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Raises if the checkpoint predates the current observation schema.
        self.net, self.ckpt = load_checkpoint(path, self.device)
        self.net.eval()
        self.deterministic = deterministic
        self.path = path

    def describe(self) -> str:
        step = self.ckpt.get("global_step", "?")
        return (f"{self.path}  step {step}  device {self.device}  "
                f"{'argmax' if self.deterministic else 'sampled'}")

    def __call__(self, obs: dict) -> Action:
        torch = self._torch

        def t(a) -> "torch.Tensor":
            arr = np.asarray(a, dtype=np.float32).reshape(-1)
            return torch.as_tensor(arr, device=self.device).unsqueeze(0)

        with torch.no_grad():
            play, slot, tile, *_ = self.net.act(
                t(self._flatten(obs)),
                t(obs["hand_playable"]),
                t(obs["hand_is_spell"]),
                t(obs["playable_mask"]),
                deterministic=self.deterministic,
            )
        return self._to_action(int(play[0]), int(slot[0]), int(tile[0]))


def make_policy(args):
    if args.policy:
        policy = CheckpointPolicy(args.policy, deterministic=not args.sample)
        print(f"policy: {policy.describe()}")
        return policy
    print("policy: random baseline")
    return RandomSimPolicy(seed=args.seed)


#: Named groups accepted by ``--opponent``. ``baseline`` is the frozen four
#: every recorded win rate was measured against; ``punishers`` is the
#: placement-punishing pair, which has no recorded baseline yet. Keeping them
#: separately addressable is what stops the two being averaged together into
#: a number that means nothing.
OPPONENT_GROUPS: dict[str, tuple[str, ...]] = {
    "baseline": BASELINE_POOL,
    "punishers": PUNISHER_POOL,
}


def parse_opponents(spec: str) -> tuple[str, ...]:
    """A group name, ``all``, or a comma-separated list.

    Episodes round-robin through whatever comes back.
    """
    if spec == "all":
        # Every real archetype. ``idle`` plays nothing, so it is only useful
        # when asked for by name.
        return tuple(n for n in sorted(OPPONENTS) if n != "idle")
    if spec in OPPONENT_GROUPS:
        return OPPONENT_GROUPS[spec]
    names = tuple(n.strip() for n in spec.split(",") if n.strip())
    unknown = [n for n in names if n not in OPPONENTS]
    if unknown:
        raise SystemExit(
            f"Unknown opponent(s) {unknown}. Have: {sorted(OPPONENTS)}, "
            f"a group in {sorted(OPPONENT_GROUPS)}, "
            f"or 'all' for every archetype except idle."
        )
    if not names:
        raise SystemExit("--opponent needs at least one name.")
    return names


def make_env(args, opponent: str, index: int = 0) -> SimEnv:
    noise = ObservationNoise.off() if args.no_noise else ObservationNoise()
    seed = None if args.seed is None else args.seed + index
    return SimEnv(
        seed=seed,
        randomize_scale=0.0 if args.no_randomize else args.randomize,
        noise=noise,
        levels=LevelSpread.off() if args.no_levels else LevelSpread(
            troop_spread=args.level_spread, tower_spread=args.level_spread
        ),
        decks=DeckSpread.off() if args.no_decks else DeckSpread(),
        # Structured decks are opponent-only and opt-in: turning them on for
        # the frozen four would change the episodes their recorded numbers
        # were measured on.
        opponent_decks=ArchetypeDeckSpread() if args.structured_decks else None,
        opponent=make_opponent(opponent, seed=seed),
    )


def run_headless(args) -> None:
    policy = make_policy(args)
    names = parse_opponents(args.opponent)
    tally: dict[str, dict[str, int]] = {n: {} for n in names}
    total_steps = 0
    started = time.perf_counter()

    for ep in range(args.episodes):
        name = names[ep % len(names)]
        env = make_env(args, name, ep)
        obs = env.reset()
        done = False
        reward_sum = 0.0
        while not done:
            obs, reward, done, info = env.step(policy(obs))
            reward_sum += reward
            total_steps += 1
        result = info["result"]
        tally[name][result] = tally[name].get(result, 0) + 1
        if args.verbose:
            print(
                f"ep {ep + 1:3d}  vs {name:15s} {result:5s}  "
                f"crowns {info['crowns'][0]}-{info['crowns'][1]}  "
                f"t={info['match_time']:5.1f}s  reward={reward_sum:7.2f}  "
                f"lvl f{info['levels']['friendly']}/e{info['levels']['enemy']}"
            )
        env.close()

    elapsed = time.perf_counter() - started
    print(f"\n{args.episodes} episodes in {elapsed:.1f}s")
    for name in names:
        counts = tally[name]
        played = sum(counts.values())
        wins = counts.get("win", 0)
        if played:
            print(f"  {name:16s} {100 * wins / played:5.0f}% win  "
                  f"({wins}/{played})  {counts}")
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

    from .render import RewardTrace, render

    policy = make_policy(args)
    names = parse_opponents(args.opponent)
    delay = max(1, int(1000 / args.fps))
    paused = False
    trace = RewardTrace()

    for ep in range(args.episodes):
        name = names[ep % len(names)]
        env = make_env(args, name, ep)
        obs = env.reset()
        trace.reset(opponent=name)
        done = False
        while not done:
            if not paused:
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                trace.update(reward, info, action)
            cv2.imshow("clashBot sim", render(env, trace))
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = not paused
            elif key == ord("n") and paused:
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                trace.update(reward, info, action)
        print(
            f"ep {ep + 1}: vs {name:15s} {info['result']} "
            f"{info['crowns'][0]}-{info['crowns'][1]} at "
            f"{info['match_time']:.0f}s  return {trace.total:+.2f}"
        )
        # Hold the final frame so the result is readable.
        cv2.imshow("clashBot sim", render(env, trace))
        if cv2.waitKey(args.hold) & 0xFF == ord("q"):
            break
        env.close()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="clashBot simulator")
    p.add_argument("--watch", action="store_true", help="open the visual debugger")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=float, default=12.0, help="viewer playback rate")
    p.add_argument("--hold", type=int, default=1200,
                   help="ms to hold the final frame of each match")
    p.add_argument("--policy", default=None,
                   help="RL checkpoint to watch (e.g. models/run1.pt); "
                        "default is the random baseline")
    p.add_argument("--sample", action="store_true",
                   help="sample from the policy instead of taking the argmax "
                        "(the benchmark measures argmax)")
    p.add_argument("--no-noise", action="store_true",
                   help="perfect perception (evaluation, not training)")
    p.add_argument("--no-randomize", action="store_true",
                   help="exact stat table (evaluation, not training)")
    p.add_argument("--randomize", type=float, default=1.0,
                   help="stat randomization scale")
    p.add_argument("--benchmark", action="store_true",
                   help="throughput run: 50 episodes, headless")
    p.add_argument("--opponent", default="idle",
                   help="scripted opponent(s): one name, a comma-separated "
                        "list, or 'all' for every archetype except idle. "
                        "Episodes round-robin through them. Default 'idle' "
                        "plays nothing, so any win rate against it is "
                        f"meaningless. Have: {sorted(OPPONENTS)}")
    p.add_argument("--no-levels", action="store_true",
                   help="pin all card/tower levels to tournament standard "
                        "(evaluation, not training)")
    p.add_argument("--level-spread", type=int, default=1,
                   help="+/- card and tower levels sampled per episode")
    p.add_argument("--structured-decks", action="store_true",
                   help="give the OPPONENT a role-structured deck (tank, two "
                        "spells, building, mini tank, swarm, air defense). "
                        "Opponent-only; the policy's deck is unaffected.")
    p.add_argument("--no-decks", action="store_true",
                   help="fixed default deck instead of sampling one per "
                        "episode (evaluation, not training)")
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
