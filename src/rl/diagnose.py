"""Probes that answer "is this opponent pool teaching placement?".

    python -m src.rl.diagnose placement
    python -m src.rl.diagnose placement --episodes 300 --pool punishers
    python -m src.rl.diagnose tiles models/run3.pt

``docs/rl-training.md`` §7 records that the probe scripts behind §3 and §5.2
were written to a session scratchpad and lost. This is the rebuild, kept in
the repo so a measurement can be repeated rather than remembered.

Two probes:

``placement``
    Does WHERE you place change the outcome against this pool? Two policies
    that differ only in placement - one spreading over legal tiles, one
    always using the same tile - are scored against the same episodes. The
    gap between them is how much placement is worth against that pool.

    This is the measurement that justifies the punisher pool existing. A pool
    that pays the same regardless of where cards go cannot teach placement
    however the reward is shaped.

``tiles``
    §5.2's instrument for a trained checkpoint: the DETERMINISTIC argmax tile
    distribution and the top1-minus-median logit margin. Do NOT use entropy
    for this - the benchmark plays argmax, so flat-looking logits still rank
    tiles perfectly well, and ``ent_coef`` is applied unweighted specifically
    to hold the placement heads near uniform.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import numpy as np

from ..env.actions import Action
from ..game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE
from ..sim.env import ArchetypeDeckSpread, SimEnv
from ..sim.opponents import BASELINE_POOL, OPPONENTS, PUNISHER_POOL, make_opponent

POOLS: dict[str, tuple[str, ...]] = {
    "baseline": BASELINE_POOL,
    "punishers": PUNISHER_POOL,
}

#: Where the concentrated policy always plays: in front of the friendly
#: princess towers. This is ``run3``'s own most-used tile, so the probe
#: mimics the failure mode actually observed rather than an invented one.
DEFAULT_TILE = (3, 12)


class PlacementProbePolicy:
    """Plays at a fixed rate; ``concentrated`` changes only WHERE.

    Play rate and slot choice are identical between the two modes, so the
    difference in win rate is attributable to placement and nothing else.
    """

    def __init__(self, concentrated: bool, tile: tuple[int, int],
                 no_op_prob: float = 0.7, seed: int | None = None):
        self.concentrated = concentrated
        self.tile = tile
        self.no_op_prob = no_op_prob
        # TWO streams, and the split is what makes the probe a measurement.
        # `rng` decides whether to play and with which slot; `place_rng`
        # decides only where. Drawing both from one stream would let the
        # spread mode's extra tile draw desynchronise the two policies after
        # the first placement, so they would act on different steps and the
        # delta would be part play-rate, part placement. Caught by
        # test_both_modes_play_at_the_same_rate.
        self.rng = random.Random(seed)
        self.place_rng = random.Random((seed or 0) + 991)

    def __call__(self, obs: dict) -> Action:
        if self.rng.random() < self.no_op_prob:
            return Action.no_op()
        slots = [i for i in range(HAND_SIZE) if obs["hand_playable"][i] > 0]
        if not slots:
            return Action.no_op()
        slot = self.rng.choice(slots)
        is_spell = obs["hand_is_spell"][slot] > 0
        tiles = np.argwhere(obs["playable_mask"] > 0)
        if self.concentrated:
            tx, ty = self.tile
            if is_spell or obs["playable_mask"][ty][tx] > 0:
                return Action(slot, tx, ty)
            # Fall back to the NEAREST legal tile rather than a no-op, so the
            # two modes place on the same steps. A no-op here would lower the
            # concentrated policy's play rate whenever its tile is blocked
            # and confound the comparison with aggression.
            if len(tiles) == 0:
                return Action.no_op()
            ny, nx = min(tiles, key=lambda t: (t[0] - ty) ** 2 + (t[1] - tx) ** 2)
            return Action(slot, int(nx), int(ny))
        if is_spell:
            return Action(slot, self.place_rng.randrange(ARENA_COLS),
                          self.place_rng.randrange(ARENA_ROWS))
        if len(tiles) == 0:
            return Action.no_op()
        ty, tx = tiles[self.place_rng.randrange(len(tiles))]
        return Action(slot, int(tx), int(ty))


def score(pool: tuple[str, ...], concentrated: bool, *, episodes: int,
          tile: tuple[int, int], structured: bool, seed: int) -> float:
    """Win rate of the probe policy against ``pool``, as a percentage."""
    wins = 0
    for ep in range(episodes):
        name = pool[ep % len(pool)]
        env = SimEnv(
            seed=seed + ep,
            opponent=make_opponent(name, seed=seed + ep),
            opponent_decks=ArchetypeDeckSpread() if structured else None,
        )
        obs = env.reset()
        policy = PlacementProbePolicy(concentrated, tile, seed=seed + ep)
        done = False
        info: dict = {}
        while not done:
            obs, _reward, done, info = env.step(policy(obs))
        wins += info.get("result") == "win"
    return 100.0 * wins / episodes


def probe_placement(args) -> None:
    pools = POOLS if args.pool == "all" else {args.pool: POOLS[args.pool]}
    print(
        f"placement sensitivity, n={args.episodes} per cell, "
        f"concentrated tile = c{args.tile[0]}r{args.tile[1]}\n"
    )
    print(f"  {'pool':22s} {'spread':>8s} {'concentrated':>13s} {'delta':>8s}")
    for label, pool in pools.items():
        spread = score(pool, False, episodes=args.episodes, tile=args.tile,
                       structured=args.structured_decks, seed=args.seed)
        conc = score(pool, True, episodes=args.episodes, tile=args.tile,
                     structured=args.structured_decks, seed=args.seed)
        print(f"  {label:22s} {spread:7.1f}% {conc:12.1f}% "
              f"{conc - spread:+7.1f}", flush=True)
    print(
        "\nDelta is what placement is WORTH against that pool. A pool whose "
        "\ndelta is near zero cannot teach placement, however the reward is "
        "\nshaped - the outcome simply does not depend on where cards go."
        "\n\nSample size: the SE of a difference between two ~40% rates at "
        f"n={args.episodes}"
        f"\nis about {2 * (0.24 / args.episodes) ** 0.5 * 100:.1f}pp. "
        "Do not read a delta smaller than that."
    )


def probe_tiles(args) -> None:
    """Argmax tile distribution and logit margin for a checkpoint (§5.2)."""
    import torch

    from ..env.observation import ObservationBuilder
    from .train import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ckpt = load_checkpoint(args.checkpoint, device)
    net.eval()

    pool = POOLS[args.pool] if args.pool in POOLS else (args.pool,)
    used: Counter = Counter()
    margins: list[float] = []

    for ep in range(args.episodes):
        name = pool[ep % len(pool)]
        env = SimEnv(seed=args.seed + ep,
                     opponent=make_opponent(name, seed=args.seed + ep))
        obs = env.reset()
        done = False
        while not done:
            def t(a):
                arr = np.asarray(a, dtype=np.float32).reshape(-1)
                return torch.as_tensor(arr, device=device).unsqueeze(0)

            with torch.no_grad():
                play, slot, tile, *rest = net.act(
                    t(ObservationBuilder.flatten(obs)),
                    t(obs["hand_playable"]), t(obs["hand_is_spell"]),
                    t(obs["playable_mask"]), deterministic=True,
                )
            if int(play[0]):
                idx = int(tile[0])
                used[(idx % ARENA_COLS, idx // ARENA_COLS)] += 1
                logits = rest[-1] if rest else None
                if logits is not None and hasattr(logits, "shape"):
                    row = np.asarray(logits[0].detach().cpu(), dtype=float)
                    legal = row[np.isfinite(row) & (row > -1e8)]
                    if legal.size:
                        margins.append(float(legal.max() - np.median(legal)))
            obs, _r, done, _i = env.step(
                Action(int(slot[0]), int(tile[0]) % ARENA_COLS,
                       int(tile[0]) // ARENA_COLS)
                if int(play[0]) else Action.no_op()
            )

    total = sum(used.values())
    if not total:
        print("the policy never placed a card; nothing to report")
        return
    print(f"placements: {total} over {args.episodes} episodes")
    print(f"distinct tiles used: {len(used)} of {ARENA_COLS * ARENA_ROWS}")
    top, n = used.most_common(1)[0]
    print(f"most-used tile: c{top[0]}r{top[1]} at {100.0 * n / total:.0f}%")
    if margins:
        print(f"top1-minus-median legal logit: {np.mean(margins):+.2f}")
    print("\nFor reference, docs/rl-training.md §3 measured: conv 8 tiles, "
          "72% on one,\nmargin +0.19; mlp 21 tiles, 52% on one, margin +0.92.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="probe", required=True)

    a = sub.add_parser("placement", help="is placement worth anything here?")
    a.add_argument("--episodes", type=int, default=150)
    a.add_argument("--pool", default="all",
                   choices=sorted(POOLS) + ["all"])
    a.add_argument("--tile", nargs=2, type=int, default=list(DEFAULT_TILE),
                   metavar=("X", "Y"))
    a.add_argument("--structured-decks", action="store_true")
    a.add_argument("--seed", type=int, default=4242)
    a.set_defaults(func=probe_placement)

    b = sub.add_parser("tiles", help="argmax tile spread for a checkpoint")
    b.add_argument("checkpoint")
    b.add_argument("--episodes", type=int, default=20)
    b.add_argument("--pool", default="baseline",
                   choices=sorted(POOLS) + sorted(OPPONENTS))
    b.add_argument("--seed", type=int, default=4242)
    b.set_defaults(func=probe_tiles)

    args = p.parse_args()
    if args.probe == "placement":
        args.tile = (int(args.tile[0]), int(args.tile[1]))
    args.func(args)


if __name__ == "__main__":
    main()
