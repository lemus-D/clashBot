"""Unit stats for the simulator, loaded from real Clash Royale game data.

The table is GENERATED, not written by hand - ``unit_stats.json`` comes from
``tools/derive_unit_stats.py``, which reads RoyaleAPI's cr-api-data (extracted
from the game client) and normalises everything to tournament standard
(displayed level 11).

This replaced a hand-written table, and the way that table was wrong is worth
recording. Its numbers were recalled rather than transcribed, and they turned
out to be approximately LEVEL 1 values - Knight 660 against a real 1766,
Archer damage 42 against a real 107. Level 1 is also not comparable across
rarities (a Rare's level 1 is displayed level 3), so the errors ran from 1.03x
to 2.68x depending on the card. A uniform scale error would have been nearly
harmless, since time-to-kill ratios survive it. A non-uniform one silently
distorts which units beat which, which is exactly what a policy learns.

Distances are GRID TILES: the real game's 18x32 arena halved to this project's
9x16. Times are seconds.

Two things still guard against the numbers being wrong:

1. ``confidence`` per stat. ``MEASURED`` means checked against real gameplay,
   ``WIKI`` transcribed from real game data, ``GUESS`` recalled or judged.
   Everything is WIKI now; nothing has been validated against actual play, so
   the mapping from these numbers to observed behaviour is still unproven.
2. ``randomize`` perturbs the table per episode. Reinforcement learning will
   exploit a constant it can pin down, so a stat that moves cannot become the
   foundation of a strategy that only works in here.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, replace
from enum import Enum

from ..game.cards import SPELL_CARDS
from ..game.classes import ARENA_CLASSES

STATS_PATH = os.path.join(os.path.dirname(__file__), "unit_stats.json")


class Confidence(str, Enum):
    MEASURED = "measured"  # checked against real gameplay
    WIKI = "wiki"          # transcribed from real game data
    GUESS = "guess"        # recalled or judged, unverified


class Target(str, Enum):
    GROUND = "ground"
    AIR = "air"
    BOTH = "both"
    BUILDINGS = "buildings"


@dataclass(frozen=True)
class UnitStats:
    """One arena unit. ``count`` > 1 means the card deploys a squad."""

    name: str
    hp: float
    damage: float
    hit_speed: float           # seconds between attacks
    attack_range: float        # grid tiles
    speed: float               # grid tiles/sec, 0 for buildings
    targets: Target
    flying: bool = False
    count: int = 1             # units deployed per card
    deploy_time: float = 1.0   # seconds before it can act
    splash_radius: float = 0.0
    aggro_range: float = 2.75  # grid tiles it will divert to attack within
    collision_radius: float = 0.25
    is_building: bool = False
    lifetime: float = 0.0      # buildings only; 0 = permanent
    spawns: str | None = None  # unit name this building produces
    spawn_period: float = 0.0  # seconds between spawn batches
    spawn_count: int = 1
    spawn_on_death: int = 0    # units released when destroyed
    confidence: Confidence = Confidence.WIKI

    @property
    def dps(self) -> float:
        return self.damage / self.hit_speed


def _load() -> tuple[dict[str, UnitStats], dict[str, tuple[float, float, float]]]:
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"{STATS_PATH} is missing. It is generated from real game data, "
            f"not written by hand: run `python tools/derive_unit_stats.py`."
        )
    with open(STATS_PATH, encoding="utf-8") as f:
        blob = json.load(f)

    units: dict[str, UnitStats] = {}
    for key, u in blob["units"].items():
        units[key] = UnitStats(
            name=key,
            hp=float(u["hp"]),
            damage=float(u["damage"]),
            hit_speed=float(u["hit_speed"]),
            attack_range=float(u["attack_range"]),
            speed=float(u["speed"]),
            targets=Target(u["targets"]),
            flying=bool(u["flying"]),
            count=int(u["count"]),
            deploy_time=float(u["deploy_time"]),
            aggro_range=float(u["aggro_range"]),
            collision_radius=float(u["collision_radius"]),
            is_building=bool(u["is_building"]),
            lifetime=float(u.get("lifetime", 0.0)),
            spawns=u.get("spawns"),
            spawn_period=float(u.get("spawn_period", 0.0)),
            spawn_count=int(u.get("spawn_count") or 1),
            spawn_on_death=int(u.get("spawn_on_death", 0)),
        )

    spells = {
        k: (float(s["damage"]), float(s["radius"]), float(s["tower_damage_mult"]))
        for k, s in blob["spells"].items()
    }
    return units, spells


UNIT_STATS, SPELL_DAMAGE = _load()

# Building-only attackers ignore troops entirely, so an aggro radius would
# describe nothing. Zeroed to make that explicit rather than leaving a number
# that looks like it does something.
for _name, _u in list(UNIT_STATS.items()):
    if _u.targets is Target.BUILDINGS:
        UNIT_STATS[_name] = replace(_u, aggro_range=0.0)

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
            f"here reads inputs that real play can never produce. Add the "
            f"missing units to tools/derive_unit_stats.py and regenerate."
        )


validate_against_manifest()


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------

# Fractional spread per confidence level. WIKI still moves: the numbers are
# real, but the MAPPING from them to this sim's behaviour is unvalidated, and
# balance patches move them anyway.
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
    fixed set of numbers learns timings that hold only for those numbers;
    trained across a range, it has to learn something that survives being
    wrong, which is the situation it will actually be deployed into.
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
    here, so the provenance should be visible rather than buried in a
    docstring.
    """
    by_level: dict[str, list[str]] = {}
    for name, unit in sorted(UNIT_STATS.items()):
        by_level.setdefault(unit.confidence.value, []).append(name)
    lines = [
        "Unit stat confidence "
        "(source: RoyaleAPI cr-api-data, tournament standard / level 11):"
    ]
    for level in ("measured", "wiki", "guess"):
        names = by_level.get(level, [])
        lines.append(f"  {level:9s} {len(names):2d}  {', '.join(names) or '-'}")
    lines.append(
        "  NOTE: 'wiki' means the NUMBERS are real, not that the simulation "
        "built on them has been validated against actual play."
    )
    return "\n".join(lines)
