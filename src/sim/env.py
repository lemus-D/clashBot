"""``SimEnv`` - the simulator behind the same API and schema as ``ClashEnv``.

The point of this file is that it does NOT build observations itself. It
populates a real ``GameBoard`` and a state adapter and hands them to the real
``ObservationBuilder``. There is one encoder in the project and both backends
go through it, so a policy trained here reads bytes laid out exactly the way
real play will lay them out. Duplicating the encoder would have been simpler
to write and would have drifted within a week.

What this adds on top of ``engine.Simulation``:

- a deck and hand cycle, since the engine only knows about units on a field
- observation NOISE, because the sim sees perfectly and the vision pipeline
  does not; a policy trained on clean inputs is brittle exactly where the
  real system is weakest
- reward shaping
- step CADENCE matching the real environment's measured 4 Hz
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..env.actions import Action, ActionResult
from ..env.observation import ObservationBuilder
from ..game.board import ARENA_COLS, ARENA_ROWS, GameBoard, HAND_SIZE
from ..game.cards import SPELL_CARDS, Card, Troop, get_card_cost, is_spell
from ..game.classes import ARENA_CLASSES, CARD_CLASSES
from ..game.state import TOWER_KEYS
from . import engine
from .engine import Simulation
from .opponents import OpponentView, UnitView
from .units import LEVELS, STANDARD_LEVEL, UNIT_STATS, randomize, stats_at_level

# Measured from real recordings: ClashEnv.step_period_sec is 0.25 and the
# loop holds it (median 0.251s over 2098 in-match cycles). Training at a
# different rate would teach timings that do not survive deployment.
STEP_PERIOD_SEC = 0.25
TICKS_PER_STEP = int(round(STEP_PERIOD_SEC / engine.TICK_DT))

# ----- reward -----
# Kept numerically comparable to ``env/environment.py`` so reward magnitudes
# mean roughly the same thing in both backends.
TOWER_HP_REWARD_SCALE = 1.5
TERMINAL_BASE_REWARD = 8.0
CROWN_MARGIN_REWARD = 2.0

# Explicit, on top of the HP swing that already covers the last hit. Taking a
# tower is a discrete strategic event, not just the final 3% of a bar, and
# losing one is worth flinching at.
TOWER_DESTROYED_REWARD = 3.0
TOWER_LOST_PENALTY = 3.0

# Elixir value of units killed, minus your own lost. THIS IS THE DEFENSIVE
# SIGNAL. Without it, winning a defensive exchange pays nothing at all: it
# shows up only as damage you did NOT take, and a counterfactual cannot be
# rewarded. Measured before adding it, 91% of steps produced exactly zero
# reward and 52% of an episode's entire signal was the win/loss bit at the
# end of ~700 steps.
#
# Scale is deliberately small. A 4-elixir kill pays 0.4 against ~4.5 for a
# tower, so roughly eleven clean trades equal one tower. Any larger and the
# policy would rationally farm kills instead of winning - a reward it can
# satisfy without playing the game is worse than a sparse one.
ELIXIR_TRADE_SCALE = 0.1

# Sitting on max elixir wastes regeneration. Fires from 9.5 rather than at
# the cap itself: at the cap exactly, the measurement showed it never
# triggered once in 14,400 steps, so the term was dead weight.
ELIXIR_CAP_THRESHOLD = 9.5
ELIXIR_CAP_PENALTY = 0.02
INVALID_ACTION_PENALTY = 0.05

def _mirror(tile_x: int, tile_y: int) -> tuple[int, int]:
    """Reflect a tile through the arena centre (own frame -> real coords)."""
    return ARENA_COLS - 1 - tile_x, ARENA_ROWS - 1 - tile_y


# Names a false-positive detection may take. Staged units are excluded: the
# detector has never seen them, so it cannot invent one.
_PHANTOM_NAMES: tuple[str, ...] = tuple(ARENA_CLASSES)

DEFAULT_DECK: tuple[str, ...] = (
    "knight", "archer", "minion", "goblin",
    "musketeer", "minipekka", "giant", "fireball",
)


@dataclass
class DeckSpread:
    """Per-episode deck sampling from the detectable card pool.

    A policy trained on one fixed eight learns that eight, not the game. It
    would then meet an unfamiliar hand the moment the deck changed - and the
    hand is part of the observation, so it has everything it needs to
    generalise if it is made to.

    The pool is CARD_CLASSES, so it grows on its own when the vision model
    gains cards. ``min_troops`` stops a sample that is mostly spells, which
    would be unplayable rather than merely different.
    """

    enabled: bool = True
    size: int = 8
    min_troops: int = 5

    @classmethod
    def off(cls) -> "DeckSpread":
        """Fixed DEFAULT_DECK, for evaluation runs that must be comparable."""
        return cls(enabled=False)

    def sample(self, rng: random.Random) -> tuple[str, ...]:
        if not self.enabled:
            return DEFAULT_DECK
        pool = list(CARD_CLASSES)
        size = min(self.size, len(pool))
        for _ in range(50):
            pick = rng.sample(pool, size)
            if sum(1 for c in pick if c not in SPELL_CARDS) >= self.min_troops:
                return tuple(pick)
        # Pathological pool (almost all spells): fall back rather than spin.
        return DEFAULT_DECK


@dataclass
class LevelSpread:
    """Per-episode card and tower LEVEL variation.

    Ladder play does not match players exactly: an opponent may be a level or
    two above or below you, and your own tower level moves independently of
    your card levels. Pinning everything to tournament standard would teach a
    policy exact breakpoints - "two Musketeer volleys kill a Knight" - that
    are wrong the moment the real opponent is one level off.

    The crucial part is that LEVEL IS HIDDEN. The detector reports "knight"
    with no level attached, so it is absent from the observation by
    construction and the policy CANNOT condition on it. It has to learn play
    that survives not knowing, which is exactly the real situation.

    Towers are sampled relative to their own side's troop level, because in
    the real game card levels and king level progress together but not in
    lockstep.
    """

    base: int = STANDARD_LEVEL
    troop_spread: int = 1   # +/- levels per side, around base
    tower_spread: int = 1   # +/- levels for towers, around that side's troops
    enabled: bool = True

    @classmethod
    def off(cls) -> "LevelSpread":
        """Fixed tournament standard, for evaluation runs that must be
        comparable to each other."""
        return cls(troop_spread=0, tower_spread=0, enabled=False)

    def sample(self, rng: random.Random) -> dict[str, int]:
        lo, hi = min(LEVELS), max(LEVELS)
        clamp = lambda v: max(lo, min(hi, v))
        if not self.enabled:
            b = clamp(self.base)
            return {"friendly": b, "enemy": b,
                    "friendly_tower": b, "enemy_tower": b}
        out = {}
        for side in ("friendly", "enemy"):
            troop = clamp(self.base + rng.randint(-self.troop_spread, self.troop_spread))
            out[side] = troop
            out[f"{side}_tower"] = clamp(
                troop + rng.randint(-self.tower_spread, self.tower_spread)
            )
        return out


@dataclass
class ObservationNoise:
    """How badly the simulated 'detector' sees the field.

    The real pipeline misses units behind fights, jitters box centres, and
    reports no tower bar at all when one is occluded. Perfect sim perception
    trains a policy that has never had to cope with any of that.

    Defaults are GUESSES. They should be replaced with rates measured from
    recorded detections - that is the strongest argument for storing raw
    detections in the recording format.
    """

    drop_prob: float = 0.05          # unit present but not reported
    false_positive_prob: float = 0.01  # phantom unit per frame
    position_jitter: float = 0.15    # grid tiles, gaussian sigma
    tower_stale_prob: float = 0.08   # bar occluded: previous value persists
    enabled: bool = True

    @classmethod
    def off(cls) -> "ObservationNoise":
        return cls(0.0, 0.0, 0.0, 0.0, enabled=False)


class Deck:
    """Eight cards, four in hand, the rest queued - Clash Royale's cycle.

    Playing a card sends it to the back of the queue and pulls the front of
    the queue into the vacated slot, so the same card cannot be played twice
    in a row and deck order is a real constraint the policy has to live with.
    """

    def __init__(self, cards: tuple[str, ...], rng: random.Random):
        unknown = [c for c in cards if c not in CARD_CLASSES]
        if unknown:
            raise ValueError(
                f"Deck contains cards the detector cannot recognise: "
                f"{unknown!r}. A card outside CARD_CLASSES has no observation "
                f"channel, so the policy could never see it in hand."
            )
        if len(cards) < HAND_SIZE + 1:
            raise ValueError(
                f"Deck needs more than {HAND_SIZE} cards to cycle; got "
                f"{len(cards)}."
            )
        self._order = list(cards)
        rng.shuffle(self._order)
        self.hand: list[str] = self._order[:HAND_SIZE]
        self._queue: list[str] = self._order[HAND_SIZE:]

    def play(self, slot: int) -> str:
        card = self.hand[slot]
        self._queue.append(card)
        self.hand[slot] = self._queue.pop(0)
        return card


class _SimStateAdapter:
    """Presents a ``Simulation`` through the slice of ``GameState`` that
    ``ObservationBuilder`` actually reads.

    Deliberately not a ``GameState`` subclass: that class is built around a
    wall clock and vision debouncing, neither of which exists here. This is
    the seam that lets one encoder serve two very different backends.
    """

    MATCH_MAX_DURATION = engine.MATCH_MAX_DURATION

    def __init__(self, sim: Simulation):
        self.sim = sim
        self.tower_hp: dict[str, float] = sim.tower_hp_fractions()
        self.crowns_friendly = 0
        self.crowns_enemy = 0

    def refresh(self, tower_hp: dict[str, float]) -> None:
        self.tower_hp = tower_hp
        self.crowns_friendly = self.sim.crowns[True]
        self.crowns_enemy = self.sim.crowns[False]

    def get_current_elixir(self) -> float:
        return self.sim.elixir[True]

    def get_current_match_time(self) -> float:
        return min(self.sim.time, self.MATCH_MAX_DURATION)

    def get_match_phase(self) -> str:
        return self.sim.phase()

    def is_enemy_left_alive(self) -> bool:
        return self.tower_hp["enemy_left"] > 0.0

    def is_enemy_right_alive(self) -> bool:
        return self.tower_hp["enemy_right"] > 0.0

    def is_enemy_king_active(self) -> bool:
        return self.tower_hp["enemy_king"] < 0.98


@dataclass
class SimEnv:
    """Gym-like simulator with ``ClashEnv``'s observation and action schema."""

    deck: tuple[str, ...] = DEFAULT_DECK
    opponent: object | None = None
    seed: int | None = None
    randomize_scale: float = 1.0
    noise: ObservationNoise = field(default_factory=ObservationNoise)
    levels: LevelSpread = field(default_factory=LevelSpread)
    decks: DeckSpread = field(default_factory=DeckSpread)

    sim: Simulation = field(init=False)
    board: GameBoard = field(init=False)
    state: _SimStateAdapter = field(init=False)
    step_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._builder = ObservationBuilder()
        self._episode = 0
        self._base_seed = random.randrange(1 << 30) if self.seed is None else self.seed
        self.reset()

    # ----- lifecycle -----

    def reset(self) -> dict:
        seed = self._base_seed + self._episode
        self._episode += 1
        self._rng = random.Random(seed)

        lv = self.levels.sample(self._rng)
        self.episode_levels = lv
        # Each side's table is loaded at its own level, then perturbed
        # independently - two different sources of uncertainty, both hidden
        # from the policy.
        f_units, _ = stats_at_level(lv["friendly"])
        e_units, _ = stats_at_level(lv["enemy"])
        self.sim = Simulation(
            seed=seed,
            friendly_level=lv["friendly"],
            enemy_level=lv["enemy"],
            friendly_tower_level=lv["friendly_tower"],
            enemy_tower_level=lv["enemy_tower"],
            friendly_stats=randomize(f_units, self._rng, self.randomize_scale),
            enemy_stats=randomize(e_units, self._rng, self.randomize_scale),
        )
        # Both sides draw independently - ladder does not mirror decks.
        self._deck = Deck(self.decks.sample(self._rng), self._rng)
        self._opp_deck = Deck(self.decks.sample(self._rng), self._rng)

        # Pixel dimensions are irrelevant here - the sim works in tiles and
        # never converts. GameBoard just needs non-degenerate values.
        self.board = GameBoard(ARENA_COLS * 64, ARENA_ROWS * 64)
        self.state = _SimStateAdapter(self.sim)
        self._reported_tower_hp = self.sim.tower_hp_fractions()
        self._prev_tower_hp = dict(self._reported_tower_hp)
        self._prev_destroyed: set[str] = set()
        self._prev_losses = dict(self.sim.losses)
        self.step_count = 0
        return self.observe()

    def close(self) -> None:  # parity with ClashEnv
        pass

    # ----- observation -----

    def observe(self) -> dict:
        self._populate_board()
        self.state.refresh(self._reported_tower_hp)
        return self._builder.build(self.board, self.state)

    def _populate_board(self) -> None:
        """Project sim entities onto the board the way the detector would.

        Note the collapse: ``troops_in_arena`` holds ONE troop per tile, so a
        stack of three goblins reports as one. That is a limitation of the
        shared board model rather than of the sim, and reproducing it here is
        correct - the real pipeline has exactly the same blind spot.
        """
        self.board.clear_arena()
        self.board.clear_hand()

        # What the noise model actually did this frame, for the debug view.
        # Recorded rather than inferred: a jittered unit lands on a different
        # tile than its true one, so comparing the board against truth
        # positions would report every jitter as a phantom.
        self.dropped_positions: list[tuple[float, float]] = []
        self.phantom_tiles: set[tuple[int, int]] = set()

        for slot, name in enumerate(self._deck.hand):
            self.board.cards_in_hand[slot] = Card(name)

        n = self.noise
        for e in self.sim.units():
            # Units are drawn from the moment they spawn, INCLUDING while
            # their deploy timer runs. The real detector sees the spawn
            # animation, so hiding them here would give the policy a blind
            # spot of several frames over its own placements that real play
            # does not have. They are visible but inert.
            if n.enabled and self._rng.random() < n.drop_prob:
                self.dropped_positions.append((e.x, e.y))
                continue
            x, y = e.x, e.y
            if n.enabled and n.position_jitter:
                x += self._rng.gauss(0.0, n.position_jitter)
                y += self._rng.gauss(0.0, n.position_jitter)
            tx, ty = int(x), int(y)
            if not (0 <= tx < ARENA_COLS and 0 <= ty < ARENA_ROWS):
                continue
            self.board.troops_in_arena[ty][tx] = Troop(
                e.name, "blue" if e.friendly else "red", tx, ty
            )

        if n.enabled and n.false_positive_prob and self._rng.random() < n.false_positive_prob:
            tx = self._rng.randrange(ARENA_COLS)
            ty = self._rng.randrange(ARENA_ROWS)
            if self.board.troops_in_arena[ty][tx] is None:
                # Drawn from what the DETECTOR can report, not from every unit
                # the sim models. A false positive is the detector being
                # wrong about something in its own class list; it cannot
                # hallucinate a staged card it has never been trained on.
                ghost = self._rng.choice(_PHANTOM_NAMES)
                self.board.troops_in_arena[ty][tx] = Troop(
                    ghost, self._rng.choice(("blue", "red")), tx, ty
                )
                self.phantom_tiles.add((tx, ty))

    def _update_reported_tower_hp(self) -> None:
        """Occlusion: a bar that cannot be read leaves the last value standing.

        This mirrors ``GameState``, which holds the previous reading rather
        than writing a gap, so a policy sees the same kind of stale value it
        will see in real play.
        """
        truth = self.sim.tower_hp_fractions()
        n = self.noise
        for key in TOWER_KEYS:
            stale = n.enabled and self._rng.random() < n.tower_stale_prob
            if not stale:
                self._reported_tower_hp[key] = truth[key]

    # ----- stepping -----

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        self._prev_tower_hp = dict(self.sim.tower_hp_fractions())
        self._prev_destroyed = set(self.sim.destroyed_towers)
        self._prev_losses = dict(self.sim.losses)

        result = self._apply_action(action)
        self._apply_opponent()

        for _ in range(TICKS_PER_STEP):
            self.sim.tick()
            if self.sim.finished:
                break

        self._update_reported_tower_hp()
        self.step_count += 1

        reward = self._reward(result)
        done = self.sim.finished
        obs = self.observe()
        info = {
            "step": self.step_count,
            "match_time": self.sim.time,
            "phase": self.sim.phase(),
            "result": self.sim.result,
            "crowns": (self.sim.crowns[True], self.sim.crowns[False]),
            "action_ok": result.success,
            "action_reason": result.reason,
            "reward_parts": dict(self.last_reward_parts),
            # Diagnostics only. Deliberately NOT in the observation: the
            # detector cannot read levels off the screen, so a policy that
            # could see them here would learn something it cannot use.
            "levels": dict(self.episode_levels),
            "decks": (list(self._deck._order), list(self._opp_deck._order)),
        }
        return obs, reward, done, info

    def _apply_action(self, action: Action) -> ActionResult:
        if action.is_no_op:
            return ActionResult(success=True, reason="no_op")
        if not (0 <= action.hand_index < HAND_SIZE):
            return ActionResult(success=False, reason="bad_slot")

        name = self._deck.hand[action.hand_index]
        if self.sim.elixir[True] + 1e-6 < get_card_cost(name):
            return ActionResult(success=False, reason="not_enough_elixir")
        if not self.sim.is_placeable(True, action.tile_x, action.tile_y, name=name):
            return ActionResult(success=False, reason="tile_not_placeable")

        if not self.sim.deploy(True, name, action.tile_x, action.tile_y):
            return ActionResult(success=False, reason="deploy_rejected")
        self._deck.play(action.hand_index)
        return ActionResult(success=True, reason="placed")

    def _opponent_view(self) -> OpponentView:
        """The narrow slice of state a scripted opponent may read.

        Handing over ``self`` would let an opponent inspect the policy's hand
        and elixir. A benchmark that can cheat is not a benchmark.
        """
        # Enemy (i.e. the policy's) units, mirrored into the opponent's own
        # frame so it reasons in one coordinate system throughout.
        def seen(units):
            return tuple(
                UnitView(
                    name=u.name,
                    tile_x=ARENA_COLS - 1 - int(u.x),
                    tile_y=ARENA_ROWS - 1 - int(u.y),
                    flying=u.stats.flying,
                )
                for u in units
                if u.deployed
            )

        threats = seen(self.sim.units(True))
        own_units = seen(self.sim.units(False))
        return OpponentView(
            hand=list(self._opp_deck.hand),
            elixir=self.sim.elixir[False],
            time=self.sim.time,
            phase=self.sim.phase(),
            threats=threats,
            own_units=own_units,
            can_place=lambda tx, ty, name=None: self.sim.is_placeable(
                False, *_mirror(tx, ty), name=name
            ),
        )

    def _apply_opponent(self) -> None:
        """Ask the opponent for a placement, in ITS own coordinate frame.

        The opponent reasons as though it were the friendly side and the
        placement is mirrored here - one coordinate convention in the whole
        codebase instead of two.
        """
        if self.opponent is None:
            return
        move = self.opponent(self._opponent_view())
        if move is None:
            return
        slot, tile_x, tile_y = move
        if not (0 <= slot < HAND_SIZE):
            return
        name = self._opp_deck.hand[slot]
        if self.sim.deploy(False, name, *_mirror(tile_x, tile_y)):
            self._opp_deck.play(slot)

    # ----- reward -----

    def _reward(self, action_result: ActionResult) -> float:
        parts = self.reward_parts(action_result)
        self.last_reward_parts = parts
        return sum(parts.values())

    def reward_parts(self, action_result: ActionResult) -> dict[str, float]:
        """Reward split by SOURCE, so the shaping can be audited.

        A single scalar hides whether the policy is learning from steady
        chip damage or from one win/loss bit at the end of ~700 steps. Those
        are very different learning problems and the average looks the same.
        """
        now = self.sim.tower_hp_fractions()

        def total(side: str, hp: dict[str, float]) -> float:
            return sum(v for k, v in hp.items() if k.startswith(side))

        dealt = total("enemy", self._prev_tower_hp) - total("enemy", now)
        taken = total("friendly", self._prev_tower_hp) - total("friendly", now)

        parts = {
            "tower_damage_dealt": TOWER_HP_REWARD_SCALE * dealt,
            "tower_damage_taken": -TOWER_HP_REWARD_SCALE * taken,
            "tower_destroyed": 0.0,
            "tower_lost": 0.0,
            "units_killed": ELIXIR_TRADE_SCALE * (
                self.sim.losses[False] - self._prev_losses.get(False, 0.0)
            ),
            "units_lost": -ELIXIR_TRADE_SCALE * (
                self.sim.losses[True] - self._prev_losses.get(True, 0.0)
            ),
            "elixir_cap": 0.0,
            "terminal": 0.0,
            "invalid_action": 0.0,
        }

        for key in self.sim.destroyed_towers - self._prev_destroyed:
            if key.startswith("enemy"):
                parts["tower_destroyed"] += TOWER_DESTROYED_REWARD
            else:
                parts["tower_lost"] -= TOWER_LOST_PENALTY

        if self.sim.elixir[True] >= ELIXIR_CAP_THRESHOLD:
            parts["elixir_cap"] = -ELIXIR_CAP_PENALTY

        if self.sim.finished:
            margin = self.sim.crowns[True] - self.sim.crowns[False]
            terminal = CROWN_MARGIN_REWARD * margin
            if self.sim.result == "win":
                terminal += TERMINAL_BASE_REWARD
            elif self.sim.result == "loss":
                terminal -= TERMINAL_BASE_REWARD
            parts["terminal"] = terminal

        if not action_result.success:
            parts["invalid_action"] = -INVALID_ACTION_PENALTY

        return parts

    # ----- convenience -----

    @property
    def hand(self) -> list[str]:
        return list(self._deck.hand)

    def flat_observation(self) -> np.ndarray:
        return ObservationBuilder.flatten(self.observe())
