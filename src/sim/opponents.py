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

from ..game.board import ARENA_ROWS, FRIENDLY_HALF_START_ROW
from ..game.cards import get_card_cost
from .units import UNIT_STATS, SPELL_NAMES, Target

# Lanes, in tile coordinates. The bridges sit at grid x 1.5 and 7.5.
LEFT_LANE = 1
RIGHT_LANE = 7
LANES = (LEFT_LANE, RIGHT_LANE)

# Rows in the opponent's own frame (its own half is y >= 8, own king at 14).
PUSH_ROW = 9       # just behind the river: units cross almost immediately
SUPPORT_ROW = 12   # far enough back that support walks in behind the tank
DEFEND_ROW = 11    # in front of the towers
BACKLINE_ROW = 12  # deep, so a beatdown push has room to gather support
LAST_DITCH_ROW = ARENA_ROWS - 2


@dataclass(frozen=True)
class Threat:
    """An enemy unit, in the opponent's OWN frame - already mirrored, so a
    threat inside its half reads ``tile_y >= FRIENDLY_HALF_START_ROW`` just
    like everything else the opponent reasons about."""

    name: str
    tile_x: int
    tile_y: int
    flying: bool


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
    threats: tuple[Threat, ...] = ()

    def cost(self, name: str) -> int:
        return get_card_cost(name)

    def invaders(self) -> list[Threat]:
        """Enemy units inside this opponent's own half, deepest first.

        Deepest = nearest its own king, which is the one actually about to
        do damage. Answering the closest-to-you threat instead is a common
        way to lose a tower while winning a fight somewhere irrelevant.
        """
        inside = [t for t in self.threats if t.tile_y >= FRIENDLY_HALF_START_ROW]
        return sorted(inside, key=lambda t: -t.tile_y)

    def defenders(self) -> list[int]:
        """Affordable hand slots that can actually DEFEND, cheapest first.

        Excludes spells and building-only attackers: a Giant will walk past
        the thing attacking you, so spending it on defence is worse than
        spending nothing.
        """
        out = []
        for i in self.affordable():
            stats = UNIT_STATS.get(self.hand[i])
            if stats is None or stats.targets is Target.BUILDINGS:
                continue
            out.append(i)
        return out

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


class ScriptedOpponent:
    """Base for the scripted bots: defend first, then whatever the subclass
    does for offence.

    DEFENCE IS THE POINT. Before this, none of these bots defended at all -
    they only ever pushed - and a random policy beat them 47% of the time
    simply by walking cards into an empty lane. A yardstick that cannot
    defend measures almost nothing.

    REACTION DELAY keeps them human. A bot that answers a threat on the
    frame it lands is not a hard opponent, it is an unrealistic one, and a
    policy trained against frame-perfect defence learns to beat something
    that does not exist on ladder. Each bot waits ``reaction_s`` after
    FIRST seeing an invader before it responds.
    """

    #: Seconds between an invader appearing and this bot reacting to it.
    reaction_s: float = 0.8

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._threat_since: float | None = None

    def __call__(self, view: OpponentView):
        move = self._defend(view)
        if move is not None:
            return move
        return self.attack(view)

    # ----- defence -----

    def _defend(self, view: OpponentView):
        invaders = view.invaders()
        if not invaders:
            self._threat_since = None
            return None

        # Timer starts when the FIRST invader is seen and runs until the half
        # is clear again, so a sustained push is answered once rather than
        # re-delayed by every new unit in it.
        if self._threat_since is None:
            self._threat_since = view.time
        if view.time - self._threat_since < self.reaction_s:
            return None

        slots = view.defenders()
        if not slots:
            return None

        target = invaders[0]
        slot = slots[0]  # cheapest that can do the job
        row = min(target.tile_y + 1, LAST_DITCH_ROW)
        if not view.can_place(target.tile_x, row):
            return None
        return slot, target.tile_x, row

    # ----- offence -----

    def attack(self, view: OpponentView):
        raise NotImplementedError


class Idle:
    """Plays nothing. The explicit version of ``opponent=None``.

    Useful as a floor: a policy that cannot beat Idle is broken, not weak.
    """

    name = "idle"

    def __call__(self, view: OpponentView):
        return None


class BigSpender(ScriptedOpponent):
    """Dumps the most expensive card it can afford, keeping elixir low.

    The simplest pressure test: it commits everything, immediately, with no
    thought about what it is answering. A policy should learn to punish
    over-commitment, and this is the opponent that supplies it. It now
    defends first, so the over-commitment is a choice rather than the only
    thing it knows how to do.
    """

    name = "bigspender"
    reaction_s = 1.0  # slowest to react: it would rather be spending

    def __init__(self, seed: int | None = None, spend_above: float = 5.0):
        super().__init__(seed)
        self.spend_above = spend_above

    def attack(self, view: OpponentView):
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


class Cycler(ScriptedOpponent):
    """Spends as often as possible on the cheapest thing available.

    The opposite failure mode to BigSpender: constant small pressure and a
    hand that never stalls. Between them they bracket the two ways elixir
    can be misused. Its cheap hand also makes it the quickest to defend.
    """

    name = "cycler"
    reaction_s = 0.6  # cheap cards in hand means it can answer fast

    def attack(self, view: OpponentView):
        slots = view.affordable()
        if not slots:
            return None
        slot = slots[0]
        lane = self.rng.choice(LANES)
        row = PUSH_ROW if self.rng.random() < 0.7 else DEFEND_ROW
        if not view.can_place(lane, row):
            return None
        return slot, lane, row


class TankAndSupport(ScriptedOpponent):
    """Builds a real push: a tank placed DEEP, support behind it.

    The first opponent that plays Clash Royale rather than just spending
    elixir. It saves up, commits a tank at the BACK of its own half so the
    push has room to gather, then follows with a cheap unit behind it.

    Placing the tank deep rather than at the bridge is the beatdown pattern:
    the tank walks the length of the arena and everything played behind it
    arrives together. Dropping it at the bridge - which is what this did
    before - gets a lone tank into the enemy half with nothing supporting it.
    """

    name = "tankandsupport"
    reaction_s = 0.8

    # Elixir at which a hand holding no tank cycles a cheap card instead of
    # waiting. Without this the bot DEADLOCKS: four cheap cards in hand means
    # no tank to play, and never playing means the hand never rotates, so it
    # sits at capped elixir for the whole match. Cycling to your win
    # condition is also just what a real player does.
    CYCLE_ABOVE = 9.0

    def __init__(self, seed: int | None = None, tank_min_cost: int = 4):
        super().__init__(seed)
        self.tank_min_cost = tank_min_cost
        self._lane: int | None = None  # set while a push is being built

    def attack(self, view: OpponentView):
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
        if not view.can_place(lane, BACKLINE_ROW):
            return None
        self._lane = lane
        return slot, lane, BACKLINE_ROW

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
        a failed attempt (no elixir yet, the usual case right after paying
        for a tank) would abandon the push and make this bot behave
        identically to BigSpender.
        """
        slots = view.affordable()
        if not slots:
            return None
        slot = slots[0]
        lane = self._lane
        row = min(BACKLINE_ROW + 1, LAST_DITCH_ROW)
        if not view.can_place(lane, row):
            self._lane = None
            return None
        self._lane = None
        return slot, lane, row


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
