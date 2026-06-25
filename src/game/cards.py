"""Card data: elixir cost database and lightweight card / troop data classes.

The cost database and data classes live together because the database has
no consumers outside this module, and collapsing the boundary keeps the
import graph flat.

``EMPTY_CARD`` and ``EMPTY_TILE`` are module-level singletons used by
``GameBoard`` to fill empty hand slots and arena tiles. Reusing one object
per kind keeps per-frame allocation flat instead of allocating
``9 * 16 = 144`` ``BlankSpace`` instances every clear.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost database
# ---------------------------------------------------------------------------

DEFAULT_UNKNOWN_COST = 4

CARD_COSTS: dict[str, int] = {
    # 1 elixir
    "skeletons": 1,
    "icespirit": 1,
    "firespirit": 1,
    "electrospirit": 1,
    "healspirit": 1,
    # 2 elixir
    "icegolem": 2,
    "bats": 2,
    "spear goblins": 2,
    "spearGoblins": 2,
    "spearGoblin": 2,
    "spear_goblins": 2,
    "spearGOblins": 2,
    "spearGOblin": 2,
    "goblins": 2,
    "zap": 2,
    "thelog": 2,
    "log": 2,
    "snowball": 2,
    "giantsnowball": 2,
    "barbarianbarrel": 2,
    "rage": 2,
    # 3 elixir
    "knight": 3,
    "archers": 3,
    "minions": 3,
    "bomber": 3,
    "icewizard": 3,
    "princess": 3,
    "tombstone": 3,
    "miner": 3,
    "cannon": 3,
    "skeletonarmy": 3,
    "skeletonbarrel": 3,
    "fireball": 4,
    "goblinhut": 5,
    "guards": 3,
    "darkprince": 4,
    "dartgoblin": 3,
    "elixircollector": 6,
    "elixirgolem": 3,
    # 4 elixir
    "babydragon": 4,
    "minipekka": 4,
    "musketeer": 4,
    "valkyrie": 4,
    "hogrider": 4,
    "infernodragon": 4,
    "magicarcher": 4,
    "battleram": 4,
    "battlehealer": 4,
    "mortar": 4,
    "tornado": 3,
    "fireballcard": 4,
    "poison": 4,
    "earthquake": 3,
    "freeze": 4,
    "clone": 3,
    "mirror": 1,
    # 5 elixir
    "wizard": 5,
    "musket": 4,
    "bowler": 5,
    "hunter": 4,
    "executioner": 5,
    "infernotower": 5,
    "tesla": 4,
    "bombtower": 4,
    "barbarianhut": 6,
    "furnace": 4,
    "barbarians": 5,
    "minionhorde": 5,
    "balloon": 5,
    "prince": 5,
    "ramrider": 5,
    "rascals": 5,
    "witch": 5,
    "nightwitch": 4,
    "graveyard": 5,
    "xbow": 6,
    "lightning": 6,
    # 6+ elixir
    "giant": 5,
    "goblingiant": 6,
    "elitebarbarians": 6,
    "lavahound": 7,
    "golem": 8,
    "pekka": 7,
    "megaknight": 7,
    "electrowizard": 4,
    "electrodragon": 5,
    "royalgiant": 6,
    "royalrecruits": 7,
    "royalghost": 3,
    "royalhogs": 5,
    "threemusketeers": 9,
    "sparky": 6,
    "skeletonking": 4,
    "archerqueen": 5,
    "goldenknight": 4,
    "mightyminer": 4,
    "bandit": 3,
    "fisherman": 3,
    "berserker": 3,
    "phoenix": 4,
    "monk": 5,
    "littleprince": 3,
}


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def get_card_cost(name: str) -> int:
    """Look up the elixir cost of a card by name.

    Falls back to ``DEFAULT_UNKNOWN_COST`` and logs a warning when the name
    is not in the database. This keeps the bot running on detector classes
    that haven't been mapped yet, while making the gap visible in logs.
    """
    if not name:
        return DEFAULT_UNKNOWN_COST
    key = _normalize(name)
    if key in CARD_COSTS:
        return CARD_COSTS[key]
    logger.warning(
        "Unknown card '%s' (normalized '%s') - defaulting to cost %d",
        name,
        key,
        DEFAULT_UNKNOWN_COST,
    )
    return DEFAULT_UNKNOWN_COST


def is_known_card(name: str) -> bool:
    return _normalize(name) in CARD_COSTS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class Card:
    __slots__ = ("name", "cost")

    def __init__(self, name: str, cost: Optional[int] = None):
        self.name = name
        self.cost = cost if cost is not None else get_card_cost(name)

    def __repr__(self) -> str:
        return f"Card(name={self.name!r}, cost={self.cost})"


class Troop:
    __slots__ = ("name", "color", "tile_x", "tile_y")

    def __init__(
        self,
        name: str,
        color: str,
        tile_x: Optional[int] = None,
        tile_y: Optional[int] = None,
    ):
        self.name = name
        self.color = color
        self.tile_x = tile_x
        self.tile_y = tile_y

    def __repr__(self) -> str:
        return f"Troop(name={self.name!r}, color={self.color!r}, tile=({self.tile_x},{self.tile_y}))"


class BlankSpace:
    """Sentinel for an empty hand slot or arena tile.

    Prefer the module-level ``EMPTY_CARD`` / ``EMPTY_TILE`` singletons over
    constructing new instances in hot paths.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "BlankSpace()"


EMPTY_CARD: BlankSpace = BlankSpace()
EMPTY_TILE: BlankSpace = BlankSpace()


def is_empty(value: object) -> bool:
    return isinstance(value, BlankSpace)
