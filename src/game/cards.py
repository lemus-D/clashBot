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
    "goblins": 2,
    "speargoblins": 2,
    "arrows": 3,
    "archers": 3,
    "minions": 3,
    "knight": 3,
    "goblinhut": 4,
    "goblincage": 4,
    "musketeer": 4,
    "fireball": 4,
    "minipekka": 4,
    "giant": 5,
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
