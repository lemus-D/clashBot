"""Scripted opponents for the simulator.

These are the FIXED end of the opponent pool. Self-play and a frozen-checkpoint
league come later, but scripted bots keep a job neither of those can do:
they are a stable yardstick. Self-play win rate sits at ~50% by construction
no matter how strong or weak the policy is, so without a fixed opponent there
is no way to tell improvement from drift. Freeze these and never tune them to
beat the current policy - the moment they move, the metric stops meaning
anything.

Each opponent reasons as though it were the FRIENDLY side. ``SimEnv`` mirrors
the placement through the arena centre before deploying, so there is one
coordinate convention in the codebase instead of two.

An opponent is any callable ``(OpponentView) -> (slot, tile_x, tile_y) | None``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..game.cards import get_card_cost
from .units import SPELL_NAMES

# Lanes, in tile coordinates. The bridges sit at grid x 1.5 and 7.5.
LEFT_LANE = 1
RIGHT_LANE = 7
LANES = (LEFT_LANE, RIGHT_LANE)

# Rows in the opponent's own frame (its own half is y >= 8).
PUSH_ROW = 9      # just behind the river: units cross almost immediately
SUPPORT_ROW = 12  # far enough back that support walks in behind the tank
DEFEND_ROW = 11   # in front of the towers


@dataclass
class OpponentView:
    """Everything a scripted opponent may look at, in its own frame.

    Deliberately narrow. Handing over the whole env would let an opponent
    read the policy's hand or elixir, and a benchmark that cheats is not a
    benchmark.
    """

    hand: list[str]
    elixir: float
    time: float
    phase: str
    can_place: object  # (tile_x, tile_y) -> bool, own frame

    def cost(self, name: str) -> int:
        return get_card_cost(name)

    def affordable(self, *, exclude_spells: bool = True) -> list[int]:
        """Hand slots this opponent can pay for right now, cheapest first.

        Spells are skipped by DEFAULT because these three bots are
        deliberately simple, not because spells are unusable - they are
        castable anywhere as of the placement fix. Aiming one well needs to
        know where the enemy has clumped, which is more judgement than a
        fixed yardstick should have. Pass ``exclude_spells=False`` if a
        future opponent wants them.
        """
        slots = [
            i for i, name in enumerate(self.hand)
            if self.cost(name) <= self.elixir + 1e-6
            and not (exclude_spells and name in SPELL_NAMES)
        ]
        return sorted(slots, key=lambda i: self.cost(self.hand[i]))


class Idle:
    """Plays nothing. The explicit version of ``opponent=None``.

    Useful as a floor: a policy that cannot beat Idle is broken, not weak.
    """

    name = "idle"

    def __call__(self, view: OpponentView):
        return None


class BigSpender:
    """Dumps the most expensive card it can afford, keeping elixir low.

    The simplest possible pressure test: it commits everything, immediately,
    with no thought about what it is answering. A policy should learn to
    punish over-commitment, and this is the opponent that supplies it.
    """

    name = "bigspender"

    def __init__(self, seed: int | None = None, spend_above: float = 5.0):
        self.rng = random.Random(seed)
        self.spend_above = spend_above

    def __call__(self, view: OpponentView):
        if view.elixir < self.spend_above:
            return None
        slots = view.affordable()
        if not slots:
            return None
        slot = slots[-1]  # affordable() is cheapest-first
        lane = self.rng.choice(LANES)
        if not view.can_place(lane, PUSH_ROW):
            return None
        return slot, lane, PUSH_ROW


class Cycler:
    """Spends as often as possible on the cheapest thing available.

    The opposite failure mode to BigSpender: constant small pressure and a
    hand that never stalls. Between them they bracket the two ways elixir can
    be misused, which is most of what a beginner policy needs to learn.
    """

    name = "cycler"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def __call__(self, view: OpponentView):
        slots = view.affordable()
        if not slots:
            return None
        slot = slots[0]
        lane = self.rng.choice(LANES)
        row = PUSH_ROW if self.rng.random() < 0.7 else DEFEND_ROW
        if not view.can_place(lane, row):
            return None
        return slot, lane, row


class TankAndSupport:
    """Builds a real push: an expensive unit at the bridge, support behind it.

    This is the first opponent that plays Clash Royale rather than just
    spending elixir. It saves up, commits a tank into one lane, then follows
    with a cheap unit placed BEHIND it so the support walks in under the
    tank's protection instead of leading with its face.

    A policy that beats BigSpender and Cycler has probably only learned to
    trade efficiently. Beating this one requires answering a committed push
    in a specific lane, which is a different and much more useful skill.
    """

    name = "tankandsupport"

    # Elixir at which a hand holding no tank cycles a cheap card instead of
    # waiting. Without this the bot DEADLOCKS: four cheap cards in hand means
    # no tank to play, and never playing means the hand never rotates, so it
    # sits at capped elixir for the whole match. Cycling to your win condition
    # is also just what a real player does.
    CYCLE_ABOVE = 9.0

    def __init__(self, seed: int | None = None, tank_min_cost: int = 4):
        self.rng = random.Random(seed)
        self.tank_min_cost = tank_min_cost
        self._lane: int | None = None  # set while a push is being built

    def __call__(self, view: OpponentView):
        if self._lane is None:
            return self._start_push(view)
        return self._add_support(view)

    def _start_push(self, view: OpponentView):
        """Wait for a genuinely expensive card, then commit a lane."""
        affordable = view.affordable()
        candidates = [
            i for i in affordable
            if view.cost(view.hand[i]) >= self.tank_min_cost
        ]
        if not candidates:
            return self._cycle(view, affordable)

        # Hold until there is enough left over to follow up, or the tank
        # walks in alone and dies to the first thing that meets it.
        slot = candidates[-1]
        if view.elixir - view.cost(view.hand[slot]) < 2.0:
            return None

        lane = self.rng.choice(LANES)
        if not view.can_place(lane, PUSH_ROW):
            return None
        self._lane = lane
        return slot, lane, PUSH_ROW

    def _cycle(self, view: OpponentView, affordable: list[int]):
        """No tank in hand: dump a cheap card at the back to rotate toward one."""
        if view.elixir < self.CYCLE_ABOVE or not affordable:
            return None
        lane = self.rng.choice(LANES)
        if not view.can_place(lane, DEFEND_ROW):
            return None
        return affordable[0], lane, DEFEND_ROW

    def _add_support(self, view: OpponentView):
        """Follow the tank with a cheap unit placed behind it.

        The lane is HELD until the support actually goes down. Clearing it on
        a failed attempt (no elixir yet, the usual case right after paying for
        a tank) would abandon the push and make this bot indistinguishable
        from BigSpender.
        """
        slots = view.affordable()
        if not slots:
            return None
        slot = slots[0]
        lane = self._lane
        if not view.can_place(lane, SUPPORT_ROW):
            self._lane = None
            return None
        self._lane = None
        return slot, lane, SUPPORT_ROW


OPPONENTS: dict[str, type] = {
    "idle": Idle,
    "bigspender": BigSpender,
    "cycler": Cycler,
    "tankandsupport": TankAndSupport,
}


def make_opponent(name: str, seed: int | None = None):
    """Build a scripted opponent by name, for the CLI and training configs."""
    try:
        cls = OPPONENTS[name]
    except KeyError:
        raise ValueError(
            f"Unknown opponent {name!r}; choose one of {sorted(OPPONENTS)}."
        ) from None
    return cls() if cls is Idle else cls(seed=seed)
