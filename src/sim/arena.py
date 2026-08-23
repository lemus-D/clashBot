"""Arena geometry: tower placement, the river, bridges, and ground pathing.

Coordinates are GRID TILES in the same 9x16 space the observation uses, with
``(0, 0)`` at the top-left. Enemy towers are at low ``y``, friendly towers at
high ``y`` - the same orientation as the captured frame, so a sim position and
a detected position mean the same thing without a flip anywhere.

Positions are continuous floats; the tile a unit occupies is ``int()`` of its
position. Tile centres are at ``x + 0.5``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..game.board import ARENA_COLS, ARENA_ROWS, FRIENDLY_HALF_START_ROW
from ..game.state import TOWER_KEYS

# The river runs along the boundary between the last enemy row and the first
# friendly one. ``FRIENDLY_HALF_START_ROW`` is where friendly placement
# begins, so it is also where the water ends.
RIVER_Y = float(FRIENDLY_HALF_START_ROW)
RIVER_HALF_WIDTH = 0.5

# Ground units can only cross at the bridges. Real bridges sit near the arena
# edges; these are the centres of columns 1 and 7 in the 9-wide grid.
BRIDGE_X: tuple[float, float] = (1.5, 7.5)
BRIDGE_HALF_WIDTH = 0.6

# Tower HP, damage and geometry are NOT constants here - they scale with the
# tower's level, which the simulator varies per episode. See
# ``units.tower_combat``. Only position and identity live in this module.


@dataclass(frozen=True)
class TowerSpec:
    key: str          # matches TOWER_KEYS, so it maps straight into tower_hp
    kind: str         # "princess" | "king", the key into the stat manifest
    x: float
    y: float
    friendly: bool

    @property
    def is_king(self) -> bool:
        return self.kind == "king"


TOWERS: tuple[TowerSpec, ...] = (
    TowerSpec("enemy_left", "princess", 2.0, 3.0, False),
    TowerSpec("enemy_right", "princess", 7.0, 3.0, False),
    TowerSpec("enemy_king", "king", 4.5, 1.5, False),
    TowerSpec("friendly_left", "princess", 2.0, 13.0, True),
    TowerSpec("friendly_right", "princess", 7.0, 13.0, True),
    TowerSpec("friendly_king", "king", 4.5, 14.5, True),
)

TOWERS_BY_KEY: dict[str, TowerSpec] = {t.key: t for t in TOWERS}

# The observation's tower vector is built by iterating TOWER_KEYS, so the sim
# must define exactly those six and no others.
assert set(TOWERS_BY_KEY) == set(TOWER_KEYS), (
    f"Simulator towers {sorted(TOWERS_BY_KEY)} do not match the observation's "
    f"TOWER_KEYS {sorted(TOWER_KEYS)}."
)


def in_bounds(x: float, y: float) -> bool:
    return 0.0 <= x < ARENA_COLS and 0.0 <= y < ARENA_ROWS


def is_river(y: float) -> bool:
    return abs(y - RIVER_Y) < RIVER_HALF_WIDTH


def on_bridge(x: float) -> bool:
    return any(abs(x - bx) <= BRIDGE_HALF_WIDTH for bx in BRIDGE_X)


def nearest_bridge_x(x: float) -> float:
    return min(BRIDGE_X, key=lambda bx: abs(bx - x))


def side_of(y: float, friendly: bool) -> bool:
    """True if ``y`` is on the given side's own half of the arena."""
    return y >= RIVER_Y if friendly else y < RIVER_Y


def ground_waypoint(
    x: float, y: float, friendly: bool, goal_x: float, goal_y: float
) -> tuple[float, float]:
    """Next point a GROUND unit should walk toward.

    Ground units cannot swim, so a unit still on its own side heads for the
    nearest bridge before anything else; once across it goes straight at its
    goal. That two-leg path is the standard simplification and reproduces the
    behaviour that actually matters strategically - lanes are a real
    constraint, and troops committed to one side cannot trivially answer a
    push on the other.

    Air units bypass this entirely and fly straight at their goal.
    """
    attacking_up = friendly  # friendly units advance toward y = 0
    crossed = (y < RIVER_Y) if attacking_up else (y >= RIVER_Y)
    if crossed:
        return goal_x, goal_y

    bridge_x = nearest_bridge_x(x)
    # Aim just past the water so the unit clears it rather than stopping on
    # the boundary and oscillating.
    beyond = RIVER_Y - RIVER_HALF_WIDTH - 0.1 if attacking_up else RIVER_Y + RIVER_HALF_WIDTH + 0.1

    # Line up with the bridge mouth first, then cross.
    if abs(x - bridge_x) > BRIDGE_HALF_WIDTH * 0.5:
        approach_y = RIVER_Y + 1.0 if attacking_up else RIVER_Y - 1.0
        return bridge_x, approach_y
    return bridge_x, beyond


def blocks_ground(x: float, y: float) -> bool:
    """True where a ground unit cannot stand: the river, except on a bridge."""
    return is_river(y) and not on_bridge(x)


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def clamp_to_arena(x: float, y: float) -> tuple[float, float]:
    return (
        min(max(x, 0.0), ARENA_COLS - 1e-3),
        min(max(y, 0.0), ARENA_ROWS - 1e-3),
    )


def deploy_position(tile_x: int, tile_y: int) -> tuple[float, float]:
    """Centre of a tile, which is where a placed card actually lands."""
    return tile_x + 0.5, tile_y + 0.5


def mirror_tile(tile_x: int, tile_y: int) -> tuple[int, int]:
    """Reflect a tile through the arena centre.

    The opponent reasons about the board from its own side. Rather than give
    the enemy a mirrored coordinate system - two conventions to keep straight
    and one more place to introduce a sign error - it uses the same one and
    mirrors at the point of placement.
    """
    return ARENA_COLS - 1 - tile_x, ARENA_ROWS - 1 - tile_y
