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
from dataclasses import dataclass, field

from ..game.board import ARENA_ROWS, FRIENDLY_HALF_START_ROW
from ..game.cards import get_card_cost
from . import roles
from .units import SPELL_DAMAGE, UNIT_STATS, SPELL_NAMES, Target

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

# The bridge row in the opponent's own frame: the first row on ITS side of
# the river, so a card dropped here crosses immediately. Used for cycling.
BRIDGE_ROW = FRIENDLY_HALF_START_ROW

# Elixir of committed units by which the two lanes must differ before the
# lighter one counts as "undefended". Below this the split is noise - one
# cheap card does not make a lane open - and chasing it would make the bot
# switch lanes constantly and never build anything.
LANE_IMBALANCE_ELIXIR = 4

# A push worth less than this in the opponent's half is left to the towers.
# Answering a lone 2-elixir card with a 4-elixir one is a trade the enemy
# wins even when the fight is won, and a policy that dribbles single cheap
# units should be rewarded with nothing.
MIN_THREAT_TO_ANSWER = 3

# Bodies that must fall inside a spell's radius before it is worth casting.
# A floor, not the real test - see SPELL_TRADE_MARGIN. One body is never the
# best use of the elixir even when it is an expensive body, because a
# defending troop trades as well and is still on the field afterwards.
MIN_SPELL_HITS = 2

# The spell must catch at least this multiple of its own cost in enemy
# elixir. THIS is the real gate, and it is measured in elixir rather than
# bodies on purpose: a swarm card is several bodies for one payment, so a
# body count reads three Goblins (two elixir between them) as a bigger prize
# than one Musketeer (four). A bot gated on body count fired Fireballs at
# single Goblin placements and lost 11 percentage points of win rate against
# a random policy, playing strictly worse than the same bot with no spells.
SPELL_TRADE_MARGIN = 1.0


@dataclass(frozen=True)
class UnitView:
    """A unit on the field, in the opponent's OWN frame - already mirrored,
    so anything inside its half reads ``tile_y >= FRIENDLY_HALF_START_ROW``
    just like everything else the opponent reasons about."""

    name: str
    tile_x: int
    tile_y: int
    flying: bool


#: Enemy units are the same shape; the name reads better at the call site.
Threat = UnitView


@dataclass(frozen=True)
class Style:
    """Per-archetype knobs. Collapsing these out of the bot bodies is what
    lets a new archetype be a table entry rather than a new class."""

    name: str
    reaction_s: float
    #: Elixir held back for defence; offence only spends above this.
    reserve: float
    #: Where this style's offence lands by default.
    push_row: int
    #: Where a heavy unit starts, if the style commits one.
    tank_row: int


STYLES: dict[str, Style] = {
    # Cheap hand, answers fast, chips constantly at the bridge.
    "cycle": Style("cycle", reaction_s=0.6, reserve=0.0,
                   push_row=PUSH_ROW, tank_row=PUSH_ROW),
    # Holds elixir for defence and converts a won defence into a push.
    "control": Style("control", reaction_s=0.7, reserve=4.0,
                     push_row=PUSH_ROW, tank_row=SUPPORT_ROW),
    # Saves to near-full, then commits a tank deep so the push gathers.
    "beatdown": Style("beatdown", reaction_s=0.8, reserve=0.0,
                      push_row=PUSH_ROW, tank_row=BACKLINE_ROW),
    # Not a real archetype - a stress test. Spends everything, immediately.
    "dump": Style("dump", reaction_s=1.0, reserve=0.0,
                  push_row=PUSH_ROW, tank_row=PUSH_ROW),
}


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
    threats: tuple[UnitView, ...] = ()
    own_units: tuple[UnitView, ...] = ()
    #: Princess-tower fill fraction keyed by LANE in this opponent's frame.
    #: ``enemy_towers[LEFT_LANE]`` is the tower a push down the left lane
    #: would hit. Empty for opponents built before tower HP was exposed.
    enemy_towers: dict[int, float] = field(default_factory=dict)
    own_towers: dict[int, float] = field(default_factory=dict)
    enemy_king_hp: float = 1.0
    own_king_hp: float = 1.0

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

    def survivors(self) -> list[UnitView]:
        """Own units still standing on own ground, deepest-advanced first.

        These are what a control player counter-pushes WITH: the defenders
        that won the exchange and are now free to walk the other way.
        """
        mine = [u for u in self.own_units
                if u.tile_y >= FRIENDLY_HALF_START_ROW]
        return sorted(mine, key=lambda u: u.tile_y)

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

    # ----- reading the enemy's placement -----
    #
    # Everything below exists to make WHERE the policy puts its cards matter
    # to the outcome. A bot that only looks at its own hand cannot punish a
    # bad placement, so it cannot teach one either.

    def lane_of(self, tile_x: int) -> int:
        """Nearest lane to a column, in this opponent's frame."""
        return min(LANES, key=lambda lane: abs(lane - tile_x))

    def weakest_enemy_lane(self) -> int:
        """The lane whose enemy princess tower is closest to falling.

        Concentrating on one tower is how matches are actually closed. Two
        towers at 50% is a draw; one at 0% and one at 100% is a crown. A bot
        that picks its lane at random - which is what every existing scripted
        opponent does - spreads damage and converts pressure into nothing.
        """
        if not self.enemy_towers:
            return LANES[0]
        alive = {lane: hp for lane, hp in self.enemy_towers.items() if hp > 0.0}
        if not alive:
            return min(self.enemy_towers, key=self.enemy_towers.get)
        return min(alive, key=alive.get)

    def enemy_units_in_lane(self, lane: int) -> list[Threat]:
        """Enemy units nearer this lane than the other, anywhere on the map."""
        return [t for t in self.threats if self.lane_of(t.tile_x) == lane]

    def undefended_lane(self) -> int | None:
        """The lane the enemy has committed FEWER units to, if it is lopsided.

        This is the placement punisher. A policy that dumps everything into
        one lane - which is exactly what a policy with a frozen tile head
        does, since it plays its favourite tile over and over - leaves the
        other lane empty, and the correct answer is to attack the empty one.
        Returns ``None`` when the split is even enough that neither lane is a
        standout, so the caller falls back to tower HP.
        """
        counts = {
            lane: sum(roles.unit_elixir_value(t.name)
                      for t in self.enemy_units_in_lane(lane))
            for lane in LANES
        }
        low = min(counts, key=counts.get)
        high = max(counts, key=counts.get)
        if counts[high] - counts[low] < LANE_IMBALANCE_ELIXIR:
            return None
        return low

    def threat_value(self) -> float:
        """Total elixir the enemy has committed inside this opponent's half.

        Used to decide whether a push is worth answering with a card at all.
        Priced per BODY (``roles.unit_elixir_value``), so a Goblins card is the
        two elixir it cost rather than six - the same accounting the spell
        gate uses, and wrong in the same expensive way if it diverges.
        """
        return sum(roles.unit_elixir_value(t.name) for t in self.invaders())

    def spell_targets(self, radius: float) -> tuple[int, int, int, float] | None:
        """Best ``(tile_x, tile_y, hits, elixir_value)`` to spell, or None.

        Centres are taken at enemy units rather than swept over the whole
        arena: an optimal centre always sits inside the cluster it hits, and
        144 candidate tiles per spell per step is a cost this does not need
        to pay when there are rarely more than a dozen units on the field.

        THIS IS THE OTHER PLACEMENT PUNISHER, and the more direct of the two.
        A spell rewards the enemy for stacking units in one spot, which is
        what a policy does when it has one favourite tile.

        Clusters are ranked by ELIXIR VALUE, not by body count, and the
        difference decides whether the bot is any good. A swarm card is
        several bodies for one payment, so three Goblins on a tile are three
        hits worth two elixir total - a Fireball on them loses two elixir
        every time. Ranking by count made this bot measurably worse than
        having no spells at all; see ``roles.unit_elixir_value``.
        """
        best: tuple[int, int, int, float] | None = None
        r2 = radius * radius
        for centre in self.threats:
            caught = [
                t for t in self.threats
                if (t.tile_x - centre.tile_x) ** 2
                + (t.tile_y - centre.tile_y) ** 2 <= r2
            ]
            value = sum(roles.unit_elixir_value(t.name) for t in caught)
            if best is None or value > best[3]:
                best = (centre.tile_x, centre.tile_y, len(caught), value)
        return best

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

    #: Which entry in ``STYLES`` supplies this bot's knobs.
    style_name: str = "cycle"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.style = STYLES[self.style_name]
        self._threat_since: float | None = None

    @property
    def reaction_s(self) -> float:
        return self.style.reaction_s

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
    style_name = "dump"  # slowest to react: it would rather be spending

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
    style_name = "cycle"  # cheap cards in hand means it can answer fast

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
    style_name = "beatdown"

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
        row = self.style.tank_row
        if not view.can_place(lane, row):
            return None
        self._lane = lane
        return slot, lane, row

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
        row = min(self.style.tank_row + 1, LAST_DITCH_ROW)
        if not view.can_place(lane, row):
            self._lane = None
            return None
        self._lane = None
        return slot, lane, row


class Control(ScriptedOpponent):
    """Defends efficiently, then turns a won defence into a push.

    The hardest of the scripted set, and the one that plays most like a real
    ladder opponent. Three behaviours, in priority order:

    1. DEFEND (inherited) - answer whatever is in its half.
    2. COUNTER-PUSH - once the half is clear, the units that WON that
       defence are still standing and already paid for. Adding support
       behind them converts a defensive trade into an attack for a fraction
       of the elixir a fresh push would cost. This is the core of control
       play and the thing the other three bots cannot do at all.
    3. CHIP - only when elixir would otherwise overflow. Otherwise it sits
       on a ``reserve`` so it can always answer the next push.

    A policy that beats BigSpender has learned to punish over-commitment.
    Beating this one requires not over-committing yourself, because anything
    that survives your attack comes straight back at you.
    """

    name = "control"
    style_name = "control"

    def attack(self, view: OpponentView):
        return self._counter_push(view) or self._chip(view)

    def _counter_push(self, view: OpponentView):
        survivors = view.survivors()
        if not survivors:
            return None
        # Most advanced survivor: the one already walking the right way.
        lead = survivors[0]
        budget = view.elixir - self.style.reserve
        slots = [i for i in view.affordable() if view.cost(view.hand[i]) <= budget]
        if not slots:
            return None
        slot = slots[-1]  # heaviest that still leaves the defensive reserve
        row = min(lead.tile_y + 1, LAST_DITCH_ROW)
        if not view.can_place(lead.tile_x, row):
            return None
        return slot, lead.tile_x, row

    def _chip(self, view: OpponentView):
        """Only spend spare elixir. Sitting at the cap wastes regeneration,
        but spending the reserve means the next push goes unanswered."""
        if view.elixir < 9.0:
            return None
        slots = view.affordable()
        if not slots:
            return None
        lane = self.rng.choice(LANES)
        if not view.can_place(lane, self.style.push_row):
            return None
        return slots[0], lane, self.style.push_row


class PlacementAware(ScriptedOpponent):
    """Mixin-ish base for opponents that READ where the enemy placed.

    The four original bots are placement-blind: they pick a lane with
    ``rng.choice(LANES)`` and answer whatever walks in. That makes them a
    fair yardstick but a poor teacher, because a policy playing every card on
    one favourite tile scores exactly the same against them as a policy that
    places well. If placement does not change the outcome, no reward signal
    derived from the outcome can teach placement - which is the standing
    hypothesis in ``docs/rl-training.md`` §4.

    These bots close that gap two ways:

    1. **Spells punish clumping.** Stacking units on one tile is precisely
       what a policy with a concentrated tile head does, and a Fireball on
       the stack takes the whole push for four elixir.
    2. **Attacks punish lane imbalance.** Everything committed to one lane
       means the other is open, so that is where the push goes.

    Neither reads anything a human could not see. ``OpponentView`` still
    hides the policy's hand and elixir.
    """

    def __call__(self, view: OpponentView):
        move = self.cast_spell(view)
        if move is not None:
            return move
        move = self._defend(view)
        if move is not None:
            return move
        return self.attack(view)

    # ----- punishment 1: spells on clumps -----

    def cast_spell(self, view: OpponentView):
        """Fireball the biggest stack of enemy units, if one is worth it.

        Fires only on a POSITIVE ELIXIR TRADE: the units caught must be worth
        at least ``SPELL_TRADE_MARGIN`` times what the spell costs. Body count
        alone is not the test - three Goblins are three bodies worth two
        elixir, and Fireballing them loses two every time. Measured, a
        count-based rule cost 11pp of win rate and made the bot strictly worse
        than having no spells at all.

        ``MIN_SPELL_HITS`` survives as a floor because a spell that lands on
        one body is never the best use of the elixir even when that body is
        expensive - a defending troop trades better and stays on the field.
        """
        slots = [
            i for i, name in enumerate(view.hand)
            if name in SPELL_NAMES and view.cost(name) <= view.elixir + 1e-6
        ]
        if not slots:
            return None
        # Heaviest affordable spell: Fireball kills what Arrows only chips.
        slot = max(slots, key=lambda i: SPELL_DAMAGE[view.hand[i]][0])
        radius = SPELL_DAMAGE[view.hand[slot]][1]
        target = view.spell_targets(radius)
        if target is None:
            return None
        tx, ty, hits, value = target
        if hits < MIN_SPELL_HITS:
            return None
        if value < view.cost(view.hand[slot]) * SPELL_TRADE_MARGIN:
            return None
        # The card NAME must go through: placement legality exempts spells
        # from the own-half rule, and without it every clump on the enemy's
        # side of the river - which is most of them - reads as illegal.
        if not view.can_place(tx, ty, view.hand[slot]):
            return None
        return slot, tx, ty

    # ----- punishment 2: attack where they are not -----

    def target_lane(self, view: OpponentView) -> int:
        """Where to push: the open lane if there is one, else the weak tower.

        Lane imbalance takes priority over tower HP because it is the more
        immediate mistake. A tower at 40% is a standing invitation, but a
        lane with the enemy's whole army in the other half is an open door
        right now, and doors close.
        """
        open_lane = view.undefended_lane()
        if open_lane is not None:
            return open_lane
        return view.weakest_enemy_lane()

    # ----- defence that does not over-answer -----

    def _defend(self, view: OpponentView):
        """As the base bot, but ignores threats too small to be worth a card.

        Over-answering is itself a placement lesson. A policy that trickles
        one cheap unit at a time gets nothing back from these bots, so the
        trickle shows up in the reward as spent elixir and no damage.
        """
        if view.threat_value() < MIN_THREAT_TO_ANSWER:
            self._threat_since = None
            return None
        return super()._defend(view)


class Punisher(PlacementAware):
    """Beatdown that builds real pushes and punishes where you place.

    Combines what ``TankAndSupport`` and ``Control`` each do separately -
    push construction from the first, defensive discipline and counter-push
    from the second - and adds the placement reading neither has.

    Priority order per step: spell a clump, defend anything worth defending,
    counter-push with whatever survived, then build. Building means the
    HEAVIEST TROOP IN HAND at the backline of the target lane, support behind
    it once it is down, and cheap cards cycled at the bridge while waiting.

    The tank rule is relative (``roles.heaviest``) rather than "wait for
    Giant". A deck is 8 of 12 cards and a hand 4 of those 8, so a bot holding
    out for one specific card would spend much of the match unable to spend
    at all. Leading with Knight when Giant is not in hand is what a human
    does in that spot.
    """

    name = "punisher"
    style_name = "beatdown"

    #: Elixir above which it cycles a cheap card rather than sit at the cap.
    CYCLE_ABOVE = 8.0

    #: Elixir kept back after paying for the tank, so support can follow.
    SUPPORT_RESERVE = 2.0

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self._lane: int | None = None

    def attack(self, view: OpponentView):
        # Support first: a tank already committed and waiting for its follow-up
        # is the most urgent thing on the board. Letting the counter-push
        # branch pre-empt it would leave the tank walking in alone, which is
        # the exact failure TankAndSupport's held lane was written to avoid.
        return (self._add_support(view)
                or self._counter_push(view)
                or self._start_push(view)
                or self._cycle(view))

    def _counter_push(self, view: OpponentView):
        """Convert a won defence into an attack, as ``Control`` does.

        The units that just won a defensive fight are already paid for and
        already walking. Support behind them costs a fraction of a fresh
        push, and this is the single most efficient thing a control-leaning
        deck does.
        """
        if view.invaders():
            return None
        survivors = view.survivors()
        if not survivors:
            return None
        lead = survivors[0]
        budget = view.elixir - self.style.reserve
        slots = [i for i in view.affordable() if view.cost(view.hand[i]) <= budget]
        if not slots:
            return None
        slot = slots[-1]
        row = min(lead.tile_y + 1, LAST_DITCH_ROW)
        if not view.can_place(lead.tile_x, row):
            return None
        return slot, lead.tile_x, row

    def _start_push(self, view: OpponentView):
        """Commit the heaviest troop in hand, deep, in the target lane."""
        if self._lane is not None:
            return None
        slots = view.affordable()
        if not slots:
            return None
        names = [view.hand[i] for i in slots]
        lead = roles.heaviest(names)
        if lead is None:
            return None
        slot = slots[names.index(lead)]
        # Hold until support is affordable too, or the tank walks in alone.
        if view.elixir - view.cost(lead) < self.SUPPORT_RESERVE:
            return None
        lane = self.target_lane(view)
        row = self.style.tank_row
        if not view.can_place(lane, row):
            return None
        self._lane = lane
        return slot, lane, row

    def _add_support(self, view: OpponentView):
        """Follow a committed tank with the cheapest thing available."""
        if self._lane is None:
            return None
        slots = view.affordable()
        if not slots:
            return None
        lane = self._lane
        row = min(self.style.tank_row + 1, LAST_DITCH_ROW)
        if not view.can_place(lane, row):
            self._lane = None
            return None
        self._lane = None
        return slots[0], lane, row

    def _cycle(self, view: OpponentView):
        """Rotate the hand by chipping at the bridge, not at the back.

        Placing the cheapest card at the bridge rather than behind the towers
        makes the cycle cost something to ignore: the card walks in and has
        to be answered. It also feeds elixir to an opponent who answers it
        well, which is the trade being offered on purpose - a policy that
        does not answer chip damage should lose a tower to it.
        """
        if view.elixir < self.CYCLE_ABOVE:
            return None
        slots = view.affordable()
        if not slots:
            return None
        lane = self.target_lane(view)
        if not view.can_place(lane, BRIDGE_ROW):
            return None
        return slots[0], lane, BRIDGE_ROW


class ControlPlus(PlacementAware):
    """``Control``'s plan, with spells and lane selection added.

    A SEPARATE opponent rather than an edit to ``Control``. The scripted pool
    is the only absolute scale this project has, and every win rate in
    ``docs/rl-training.md`` §1 was measured against the four bots exactly as
    they are; changing one of them retroactively invalidates all of it. So
    ``control`` stays frozen and this stands beside it.

    Differences from ``Control``:

    - Fireballs clumps (inherited from ``PlacementAware``).
    - Chips down the open or weaker lane instead of a random one.
    - Declines to answer threats below ``MIN_THREAT_TO_ANSWER``.
    """

    name = "controlplus"
    style_name = "control"

    def attack(self, view: OpponentView):
        return self._counter_push(view) or self._chip(view)

    def _counter_push(self, view: OpponentView):
        survivors = view.survivors()
        if not survivors:
            return None
        lead = survivors[0]
        budget = view.elixir - self.style.reserve
        slots = [i for i in view.affordable() if view.cost(view.hand[i]) <= budget]
        if not slots:
            return None
        slot = slots[-1]
        row = min(lead.tile_y + 1, LAST_DITCH_ROW)
        if not view.can_place(lead.tile_x, row):
            return None
        return slot, lead.tile_x, row

    def _chip(self, view: OpponentView):
        if view.elixir < 9.0:
            return None
        slots = view.affordable()
        if not slots:
            return None
        lane = self.target_lane(view)
        if not view.can_place(lane, self.style.push_row):
            return None
        return slots[0], lane, self.style.push_row


OPPONENTS: dict[str, type] = {
    "idle": Idle,
    "bigspender": BigSpender,
    "cycler": Cycler,
    "tankandsupport": TankAndSupport,
    "control": Control,
    "punisher": Punisher,
    "controlplus": ControlPlus,
}

#: The original four, frozen. Every win rate in docs/rl-training.md §1 was
#: measured against exactly these, so they must not change.
BASELINE_POOL: tuple[str, ...] = (
    "bigspender", "control", "cycler", "tankandsupport",
)

#: The placement-punishing pool. Scored separately; not comparable to the
#: numbers recorded against BASELINE_POOL.
PUNISHER_POOL: tuple[str, ...] = ("punisher", "controlplus")


def make_opponent(name: str, seed: int | None = None):
    """Build a scripted opponent by name, for the CLI and training configs."""
    try:
        cls = OPPONENTS[name]
    except KeyError:
        raise ValueError(
            f"Unknown opponent {name!r}; choose one of {sorted(OPPONENTS)}."
        ) from None
    return cls() if cls is Idle else cls(seed=seed)
