"""The benchmark: win rate against each frozen scripted opponent.

Two rules make this a measurement rather than a number:

1. FIXED SEED, separate from any training seed. The same checkpoint scored
   twice gives the same answer, and two checkpoints are scored on the same
   episodes.
2. The SAME distribution as training - sampled decks, levels, noise. A
   benchmark on one fixed deck with perfect perception measures a slice the
   policy will never actually play. ``clean=True`` pins all of that off when
   you specifically want the narrow, low-variance number.

The scripted opponents must stay frozen for any of this to mean anything.
Self-play win rate sits at ~50% by construction, so these are the only
absolute scale available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..sim.env import DeckSpread, LevelSpread, ObservationNoise
from ..sim.opponents import OPPONENTS
from .policy import ActorCritic, to_action
from .vec_env import VecSimEnv

#: Distinct from training seeds so evaluation episodes are never ones the
#: policy was trained on.
EVAL_SEED = 777_000

#: THE FROZEN BENCHMARK. Every win rate in docs/rl-training.md §1 was measured
#: against exactly these five, so this tuple must not gain or lose a member -
#: a number scored against a different pool is not comparable to a recorded
#: one, however similar the pools look.
BENCHMARK = ("idle", "bigspender", "control", "cycler", "tankandsupport")

#: The placement-punishing pool, scored SEPARATELY. These bots read where the
#: policy places and exploit it (spells on clumps, pushes down the lane it
#: left open), which the frozen four do not do at all. Results here start
#: from no baseline: random and any existing checkpoint must be re-scored
#: against this pool before a number against it means anything.
PUNISHER_BENCHMARK = ("idle", "controlplus", "punisher")

#: Named pools for the CLI. Every registered opponent must appear in at least
#: one of these, so a bot cannot be added and then silently never scored.
BENCHMARKS: dict[str, tuple[str, ...]] = {
    "baseline": BENCHMARK,
    "punisher": PUNISHER_BENCHMARK,
}


@dataclass
class OpponentResult:
    opponent: str
    episodes: int
    wins: int
    losses: int
    draws: int
    crowns_for: float
    crowns_against: float
    mean_return: float
    mean_length: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0


def _make_vec(opponent: str, n: int, seed: int, clean: bool) -> VecSimEnv:
    kw = dict(num_envs=n, seed=seed, opponents=(opponent,))
    if clean:
        kw.update(randomize_scale=0.0, noise=ObservationNoise.off(),
                  levels=LevelSpread.off(), decks=DeckSpread.off())
    return VecSimEnv(**kw)


@torch.no_grad()
def evaluate_opponent(
    net: ActorCritic,
    device: torch.device,
    opponent: str,
    episodes: int = 40,
    seed: int = EVAL_SEED,
    clean: bool = False,
    deterministic: bool = True,
    batch: int = 16,
) -> OpponentResult:
    """Play ``episodes`` matches against one opponent and score them.

    BATCHED. Stepping one env at a time means a GPU round trip per 0.25s of
    simulated game, and a full benchmark is ~100k of them - slow enough that
    it dominated training wall-clock when it ran one match at a time.
    Batching costs nothing in fidelity: the envs are independent.
    """
    n = max(1, min(batch, episodes))
    envs = _make_vec(opponent, n, seed, clean)
    net.eval()

    done_eps: list[dict] = []
    obs_np, masks = envs.observe()
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

    # Bounded so a policy that somehow never finishes a match cannot hang a
    # training run. 1300 steps is just past the 300s match cap at 4 Hz.
    max_steps = 1300 * (episodes // n + 2)
    for _ in range(max_steps):
        if len(done_eps) >= episodes:
            break
        play, slot, tile, *_ = net.act(
            t(obs_np), t(masks.hand_playable), t(masks.hand_is_spell),
            t(masks.playable), deterministic=deterministic,
        )
        actions = [
            to_action(int(p), int(s), int(ti))
            for p, s, ti in zip(play.tolist(), slot.tolist(), tile.tolist())
        ]
        _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info:
                done_eps.append(info["episode"])
        obs_np, masks = envs.observe()

    envs.close()
    net.train()

    done_eps = done_eps[:episodes]
    got = len(done_eps) or 1
    return OpponentResult(
        opponent=opponent,
        episodes=len(done_eps),
        wins=sum(e["result"] == "win" for e in done_eps),
        losses=sum(e["result"] == "loss" for e in done_eps),
        draws=sum(e["result"] == "draw" for e in done_eps),
        crowns_for=sum(e["crowns_for"] for e in done_eps) / got,
        crowns_against=sum(e["crowns_against"] for e in done_eps) / got,
        mean_return=float(np.mean([e["return"] for e in done_eps])) if done_eps else 0.0,
        mean_length=float(np.mean([e["length"] for e in done_eps])) if done_eps else 0.0,
    )


def evaluate(
    net: ActorCritic,
    device: torch.device,
    opponents: tuple[str, ...] = BENCHMARK,
    episodes: int = 40,
    seed: int = EVAL_SEED,
    clean: bool = False,
    deterministic: bool = True,
    batch: int = 16,
) -> list[OpponentResult]:
    unknown = [o for o in opponents if o not in OPPONENTS]
    if unknown:
        raise ValueError(f"Unknown opponents {unknown!r}; have {sorted(OPPONENTS)}.")
    return [
        evaluate_opponent(net, device, o, episodes, seed, clean, deterministic,
                          batch)
        for o in opponents
    ]


def format_results(results: list[OpponentResult]) -> str:
    lines = [
        f"  {'opponent':16s} {'win%':>5s} {'W':>4s} {'L':>4s} {'D':>4s} "
        f"{'crowns f-e':>11s} {'return':>8s} {'len':>6s}"
    ]
    for r in results:
        lines.append(
            f"  {r.opponent:16s} {100 * r.win_rate:5.0f} {r.wins:4d} "
            f"{r.losses:4d} {r.draws:4d} "
            f"{r.crowns_for:5.2f}-{r.crowns_against:<5.2f} "
            f"{r.mean_return:8.2f} {r.mean_length:6.0f}"
        )
    scored = [r for r in results if r.opponent != "idle"]
    if scored:
        overall = sum(r.wins for r in scored) / sum(r.episodes for r in scored)
        lines.append(f"  {'OVERALL (ex-idle)':16s} {100 * overall:5.0f}")
    return "\n".join(lines)


def headline(results: list[OpponentResult]) -> float:
    """Single number for tracking progress: win rate over every opponent
    except ``idle``, which any working policy beats and which therefore
    carries no signal."""
    scored = [r for r in results if r.opponent != "idle"]
    if not scored:
        return 0.0
    return sum(r.wins for r in scored) / sum(r.episodes for r in scored)
