"""Unit stats for the simulator, and the ranges they may be randomized over.

EVERY NUMBER HERE IS APPROXIMATE. They are hand-transcribed tournament-level
values, not measured against the real game, and several (aggro radius, deploy
time, spawner periods) are educated guesses. That matters more than usual:
reinforcement learning will find and exploit whatever this file says, so a
number that is confidently wrong gets baked into the policy and then fails on
transfer.

Two defences:

1. ``confidence`` marks how much each stat is trusted. ``MEASURED`` means it
   came from real observation, ``WIKI`` from transcribed game data, ``GUESS``
   from nothing better than judgement. Nothing is MEASURED yet.
2. ``randomize`` perturbs the table per-episode. A stat the policy cannot
   pin down cannot be exploited to the decimal, so it has to learn something
   robust instead. Spread scales with how little the stat is trusted.

Distances are in GRID TILES - this project's 9x16 arena, which is half the
resolution of the real game's 18x32. Wiki numbers in "tiles" are therefore
HALVED here; the raw value is kept in a comment so it stays auditable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum

from ..game.cards import SPELL_CARDS
from ..game.classes import ARENA_CLASSES


class Confidence(str, Enum):
    MEASURED = "measured"  # checked against real gameplay
    WIKI = "wiki"          # transcribed game data, not verified here
    GUESS = "guess"        # judgement call, most likely wrong


class Target(str, Enum):
    GROUND = "ground"
    AIR = "air"
    BOTH = "both"
    BUILDINGS = "buildings"


# Movement speeds in grid tiles/sec. Wiki values are CR tiles/sec: slow 0.71,
# medium 1.0, fast 1.33, very fast 1.5.
SPEED_SLOW = 0.355
SPEED_MEDIUM = 0.5
SPEED_FAST = 0.665
SPEED_VERY_FAST = 0.75

# Melee "range" is not zero - a unit stops when its hitbox touches. Half a
# grid tile is the stand-off that keeps two melee units from occupying the
# same point.
MELEE_RANGE = 0.5


@dataclass(frozen=True)
class UnitStats:
    """One arena unit. ``count`` > 1 means the card deploys a squad."""

    name: str
    hp: float
    damage: float
    hit_speed: float          # seconds between attacks
    attack_range: float       # grid tiles, MELEE_RANGE for melee
    speed: float              # grid tiles/sec, 0 for buildings
    targets: Target
    flying: bool = False
    count: int = 1            # units deployed per card
    deploy_time: float = 1.0  # seconds before it can act
    splash_radius: float = 0.0
    aggro_range: float = 2.75  # grid tiles it will divert to attack within
    is_building: bool = False
    lifetime: float = 0.0     # buildings only; 0 = permanent
    spawns: str | None = None       # unit name this building produces
    spawn_period: float = 0.0       # seconds between spawns
    spawn_count: int = 1
    spawn_on_death: int = 0         # units released when destroyed
    confidence: Confidence = Confidence.WIKI

    @property
    def dps(self) -> float:
        return self.damage / self.hit_speed


# ---------------------------------------------------------------------------
# The table. Keyed by the detector's own normalized class names, so it cannot
# describe a unit the vision pipeline could never report - see
# ``validate_against_manifest``.
# ---------------------------------------------------------------------------

UNIT_STATS: dict[str, UnitStats] = {
    "knight": UnitStats(
        name="knight", hp=660, damage=75, hit_speed=1.2,
        attack_range=MELEE_RANGE, speed=SPEED_MEDIUM, targets=Target.GROUND,
    ),
    "archer": UnitStats(
        name="archer", hp=125, damage=42, hit_speed=1.2,
        attack_range=2.5,  # 5.0 CR tiles
        speed=SPEED_MEDIUM, targets=Target.BOTH, count=2,
    ),
    "minion": UnitStats(
        name="minion", hp=90, damage=39, hit_speed=1.0,
        attack_range=1.0,  # 2.0 CR tiles
        speed=SPEED_FAST, targets=Target.BOTH, flying=True, count=3,
    ),
    "goblin": UnitStats(
        name="goblin", hp=79, damage=47, hit_speed=1.1,
        attack_range=MELEE_RANGE, speed=SPEED_VERY_FAST,
        targets=Target.GROUND, count=3,
    ),
    "speargoblin": UnitStats(
        name="speargoblin", hp=52, damage=32, hit_speed=1.7,
        attack_range=2.5,  # 5.0 CR tiles
        speed=SPEED_VERY_FAST, targets=Target.BOTH, count=3,
    ),
    "musketeer": UnitStats(
        name="musketeer", hp=340, damage=100, hit_speed=1.1,
        attack_range=3.0,  # 6.0 CR tiles
        speed=SPEED_MEDIUM, targets=Target.BOTH,
    ),
    "minipekka": UnitStats(
        name="minipekka", hp=600, damage=325, hit_speed=1.8,
        attack_range=MELEE_RANGE, speed=SPEED_FAST, targets=Target.GROUND,
    ),
    "giant": UnitStats(
        name="giant", hp=2000, damage=126, hit_speed=1.5,
        attack_range=MELEE_RANGE, speed=SPEED_SLOW, targets=Target.BUILDINGS,
        # Building-targeters ignore everything else, so aggro range is
        # meaningless for them - it is set to 0 to make that explicit rather
        # than leaving a value that looks like it does something.
        aggro_range=0.0,
    ),
    # Goblin Brawler is spawn-only: it comes out of a dying Goblin Cage and
    # has no card. It is the reason ARENA_CLASSES and CARD_CLASSES are
    # separate lists.
    "goblinbrawler": UnitStats(
        name="goblinbrawler", hp=800, damage=159, hit_speed=1.4,
        attack_range=MELEE_RANGE, speed=SPEED_MEDIUM, targets=Target.GROUND,
        confidence=Confidence.GUESS,
    ),
    "goblincage": UnitStats(
        name="goblincage", hp=720, damage=0, hit_speed=1.0,
        attack_range=0.0, speed=0.0, targets=Target.GROUND,
        is_building=True, lifetime=30.0,
        spawns="goblinbrawler", spawn_on_death=1,
        confidence=Confidence.GUESS,
    ),
    "goblinhut": UnitStats(
        name="goblinhut", hp=800, damage=0, hit_speed=1.0,
        attack_range=0.0, speed=0.0, targets=Target.GROUND,
        is_building=True, lifetime=30.0,
        spawns="speargoblin", spawn_period=4.9, spawn_count=1,
        confidence=Confidence.GUESS,
    ),
}

# Spells are not units - they resolve instantly at a point and then are gone.
# They still appear in ARENA_CLASSES because the detector sees their visual
# effect on the field.
SPELL_DAMAGE: dict[str, tuple[float, float, float]] = {
    # name: (damage, radius in grid tiles, damage multiplier vs buildings)
    "arrows": (144, 2.0, 0.35),    # 4.0 CR tiles
    "fireball": (325, 1.25, 0.35),  # 2.5 CR tiles
}

SPELL_NAMES = frozenset(SPELL_DAMAGE)

if SPELL_NAMES != SPELL_CARDS:
    raise ValueError(
        f"Simulator spell effects {sorted(SPELL_NAMES)} do not match the "
        f"card table's SPELL_CARDS {sorted(SPELL_CARDS)}. Placement legality "
        f"keys off SPELL_CARDS while damage keys off this table, so a "
        f"mismatch means a card is castable anywhere and does nothing, or "
        f"does damage but cannot be aimed."
    )


def validate_against_manifest() -> None:
    """Every arena class must be either a unit or a spell, and vice versa.

    The simulator has to cover exactly what the detector can see. A unit the
    sim knows but the detector cannot report is a policy input that will be
    permanently zero on real hardware - a silent transfer failure, which is
    the specific thing this check exists to prevent.
    """
    described = set(UNIT_STATS) | SPELL_NAMES
    known = set(ARENA_CLASSES)
    missing = sorted(known - described)
    extra = sorted(described - known)
    if missing or extra:
        raise ValueError(
            f"Simulator unit table does not match ARENA_CLASSES: no stats "
            f"for {missing!r}, stats for non-existent {extra!r}. The sim must "
            f"model exactly what the detector can see, or a policy trained "
            f"here reads inputs that real play can never produce."
        )


validate_against_manifest()


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------

# Fractional spread per confidence level. A GUESS moves a lot because we have
# no reason to believe the written value; WIKI still moves, because
# transcription errors and balance patches are both real.
_SPREAD: dict[Confidence, float] = {
    Confidence.MEASURED: 0.02,
    Confidence.WIKI: 0.08,
    Confidence.GUESS: 0.25,
}

# Stats worth perturbing. Deliberately excludes count, flags and names -
# randomizing how many goblins a Goblin card spawns would not model
# uncertainty, it would model a different game.
_RANDOMIZED_FIELDS = (
    "hp", "damage", "hit_speed", "attack_range", "speed",
    "deploy_time", "aggro_range", "lifetime", "spawn_period",
)


def randomize(
    stats: dict[str, UnitStats],
    rng: random.Random,
    scale: float = 1.0,
) -> dict[str, UnitStats]:
    """Return a per-episode perturbed copy of the stat table.

    ``scale`` multiplies every spread: 0.0 gives the table unchanged (use it
    for evaluation, so runs are comparable), 1.0 the configured spread.

    Randomizing is not noise for its own sake. A policy trained against one
    fixed set of wrong numbers learns timings that hold only for those
    numbers; trained across a range, it has to learn something that survives
    being wrong, which is the situation it will actually be deployed into.
    """
    if scale <= 0.0:
        return dict(stats)

    out: dict[str, UnitStats] = {}
    for name, unit in stats.items():
        spread = _SPREAD[unit.confidence] * scale
        changes = {}
        for field in _RANDOMIZED_FIELDS:
            value = getattr(unit, field)
            if value:  # leave zeros alone - they mean "not applicable"
                changes[field] = value * (1.0 + rng.uniform(-spread, spread))
        out[name] = replace(unit, **changes)
    return out


def stats_confidence_report() -> str:
    """Human-readable summary of how much of the table is actually trusted.

    Printed by the sim CLI. Sim fidelity is the ceiling on everything trained
    here, so "none of this is measured" should be visible rather than buried
    in a docstring.
    """
    by_level: dict[str, list[str]] = {}
    for name, unit in sorted(UNIT_STATS.items()):
        by_level.setdefault(unit.confidence.value, []).append(name)
    lines = ["Unit stat confidence:"]
    for level in ("measured", "wiki", "guess"):
        names = by_level.get(level, [])
        lines.append(f"  {level:9s} {len(names):2d}  {', '.join(names) or '-'}")
    return "\n".join(lines)
