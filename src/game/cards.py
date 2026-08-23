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


# Keys are already normalized (see ``normalize_name``) and MUST match the
# detector's ``card <name>`` classes with the prefix stripped. The model says
# "card minion", not "card minions", so these are singular: four plural keys
# (archers/goblins/minions/speargoblins) previously matched nothing and every
# one of those cards silently took DEFAULT_UNKNOWN_COST.
#
# This is exactly the model's 12 card classes. Goblin Brawler is absent on
# purpose - it spawns from Goblin Cage and is not a playable card, which is
# why the model has no ``card goblin brawler`` either.
CARD_COSTS: dict[str, int] = {
    "skeletons":1,
    "goblin": 2,
    "speargoblin": 2,
    "bomber": 2,
    "arrows": 3,
    "archer": 3,
    "cannon": 3,
    "minion": 3,
    "tombstone": 3,
    "megaminion": 3,
    "knight": 3,
    "goblinhut": 4,
    "goblincage": 4,
    "battleram": 4,
    "musketeer": 4,
    "valkyrie": 4,
    "fireball": 4,
    "minipekka": 4,
    "giant": 5,
    "barbarians": 5
}


# Unknown names already reported. A card sits in the hand for many seconds
# and is re-resolved every perception cycle, so warning per call produced
# hundreds of identical lines per match - which is how a mismatch affecting
# four of the twelve cards stayed unnoticed: the signal was there but
# unreadable. One line per distinct name is just as loud and actually gets
# read.
_warned_unknown: set[str] = set()


def get_card_cost(name: str) -> int:
    """Look up the elixir cost of a card by name.

    Falls back to ``DEFAULT_UNKNOWN_COST`` and warns ONCE per distinct
    unknown name. This keeps the bot running on detector classes that
    haven't been mapped yet, while making the gap visible in logs.

    A warning here means ``CARD_COSTS`` disagrees with the detector's
    ``card <name>`` classes, and the cost being wrong makes
    ``hand_playable`` wrong - so it is a real bug, not noise.
    """
    key = normalize_name(name)
    cost = CARD_COSTS.get(key)
    if cost is None:
        if key not in _warned_unknown:
            _warned_unknown.add(key)
            logger.warning(
                "Unknown card '%s' (normalized '%s') - defaulting to cost %d. "
                "CARD_COSTS does not match the detector's class names; "
                "hand_playable will be wrong for this card.",
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
