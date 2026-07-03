"""Card data: elixir cost database and Card / Troop data classes.

Empty hand slots and arena tiles are represented as ``None`` throughout
the codebase; there is no sentinel object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_UNKNOWN_COST = 4


def normalize_name(name: str) -> str:
    """Canonical card/troop key: lowercase, no spaces/underscores/hyphens."""
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


# Keys are already normalized (see ``normalize_name``).
CARD_COSTS: dict[str, int] = {
    "skeletons": 1,
    "icespirit": 1,
    "firespirit": 1,
    "electrospirit": 1,
    "healspirit": 1,
    "mirror": 1,
    "icegolem": 2,
    "bats": 2,
    "speargoblins": 2,
    "speargoblin": 2,
    "goblins": 2,
    "zap": 2,
    "thelog": 2,
    "log": 2,
    "snowball": 2,
    "giantsnowball": 2,
    "barbarianbarrel": 2,
    "rage": 2,
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
    "guards": 3,
    "dartgoblin": 3,
    "elixirgolem": 3,
    "tornado": 3,
    "earthquake": 3,
    "clone": 3,
    "royalghost": 3,
    "bandit": 3,
    "fisherman": 3,
    "berserker": 3,
    "littleprince": 3,
    "fireball": 4,
    "fireballcard": 4,
    "darkprince": 4,
    "babydragon": 4,
    "minipekka": 4,
    "musketeer": 4,
    "musket": 4,
    "valkyrie": 4,
    "hogrider": 4,
    "infernodragon": 4,
    "magicarcher": 4,
    "battleram": 4,
    "battlehealer": 4,
    "mortar": 4,
    "poison": 4,
    "freeze": 4,
    "hunter": 4,
    "tesla": 4,
    "bombtower": 4,
    "furnace": 4,
    "nightwitch": 4,
    "electrowizard": 4,
    "skeletonking": 4,
    "goldenknight": 4,
    "mightyminer": 4,
    "phoenix": 4,
    "goblinhut": 5,
    "wizard": 5,
    "bowler": 5,
    "executioner": 5,
    "infernotower": 5,
    "barbarians": 5,
    "minionhorde": 5,
    "balloon": 5,
    "prince": 5,
    "ramrider": 5,
    "rascals": 5,
    "witch": 5,
    "graveyard": 5,
    "giant": 5,
    "electrodragon": 5,
    "royalhogs": 5,
    "archerqueen": 5,
    "monk": 5,
    "elixircollector": 6,
    "xbow": 6,
    "lightning": 6,
    "goblingiant": 6,
    "elitebarbarians": 6,
    "royalgiant": 6,
    "sparky": 6,
    "barbarianhut": 6,
    "lavahound": 7,
    "pekka": 7,
    "megaknight": 7,
    "royalrecruits": 7,
    "golem": 8,
    "threemusketeers": 9,
}


def get_card_cost(name: str) -> int:
    """Look up the elixir cost of a card by name.

    Falls back to ``DEFAULT_UNKNOWN_COST`` and logs a warning when the name
    is not in the database. This keeps the bot running on detector classes
    that haven't been mapped yet, while making the gap visible in logs.
    """
    key = normalize_name(name)
    cost = CARD_COSTS.get(key)
    if cost is None:
        logger.warning(
            "Unknown card '%s' (normalized '%s') - defaulting to cost %d",
            name,
            key,
            DEFAULT_UNKNOWN_COST,
        )
        return DEFAULT_UNKNOWN_COST
    return cost


@dataclass
class Card:
    name: str
    cost: int | None = None  # resolved from CARD_COSTS when omitted

    def __post_init__(self) -> None:
        if self.cost is None:
            self.cost = get_card_cost(self.name)


@dataclass
class Troop:
    name: str
    color: str
    tile_x: int | None = None
    tile_y: int | None = None
