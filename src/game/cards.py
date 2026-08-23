"""Card data: elixir cost database and Card / Troop data classes.

Empty hand slots and arena tiles are represented as ``None`` throughout
the codebase; there is no sentinel object.

``normalize_name`` and the class lists themselves live in ``classes.py`` -
this module only adds the one thing the detector cannot tell us, which is
what each card costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .classes import CARD_CLASSES, normalize_name

logger = logging.getLogger(__name__)

DEFAULT_UNKNOWN_COST = 4

# Elixir costs for the detector's card classes. The KEYS are checked against
# ``CARD_CLASSES`` at import (see ``_validate_costs``) so this table cannot
# drift from the model, but the VALUES are genuine hand-maintained game data
# - a detector knows what a card looks like, not what it costs.
#
# Goblin Brawler is deliberately absent, and now structurally so: it spawns
# from Goblin Cage and is not playable, so it is in ARENA_CLASSES and not in
# CARD_CLASSES, and the check below never asks for its cost.
CARD_COSTS: dict[str, int] = {
    "goblin": 2,
    "speargoblin": 2,
    "arrows": 3,
    "archer": 3,
    "minion": 3,
    "knight": 3,
    "goblinhut": 4,
    "goblincage": 4,
    "musketeer": 4,
    "fireball": 4,
    "minipekka": 4,
    "giant": 5,
}


# Which cards are SPELLS. Like costs, the detector cannot tell us this, so it
# is hand-maintained and validated against CARD_CLASSES below.
#
# It exists because spells obey a different placement rule: a troop may only
# be deployed on your own half (plus a lane opened by a destroyed tower), but
# a spell may be cast ANYWHERE in the arena. Without this distinction the
# single ``playable_mask`` confined Fireball and Arrows to the friendly half,
# where they can never hit an enemy tower - so a policy would correctly learn
# that two of its twelve cards are useless.
SPELL_CARDS: frozenset[str] = frozenset({"arrows", "fireball"})


def is_spell(name: str) -> bool:
    return normalize_name(name) in SPELL_CARDS


def _validate_spells() -> None:
    """Spells must be real cards, or the placement rule keys off nothing."""
    unknown = sorted(SPELL_CARDS - set(CARD_CLASSES))
    if unknown:
        raise ValueError(
            f"SPELL_CARDS names cards the detector has no class for: "
            f"{unknown!r}. Either the manifest is stale (regenerate with "
            f"`python -m src.main --derive-classes`) or these are typos."
        )


def _validate_costs() -> None:
    """Every card the detector can see must have a cost, and vice versa.

    A missing cost silently makes ``hand_playable`` wrong for that card,
    which is the observation lying to the policy about what it may do. An
    extra cost is a dead entry suggesting this table describes a model other
    than the one loaded. Both are cheap to catch at import and expensive to
    notice partway through a training run.
    """
    known = set(CARD_CLASSES)
    have = set(CARD_COSTS)
    missing = sorted(known - have)
    extra = sorted(have - known)
    if missing or extra:
        raise ValueError(
            f"CARD_COSTS does not match the detector's card classes: "
            f"missing costs for {missing!r}, unknown extra entries {extra!r}. "
            f"Add the missing elixir costs by hand (the detector cannot "
            f"supply them) and drop the extras, or regenerate the manifest "
            f"with `python -m src.main --derive-classes` if the model itself "
            f"changed."
        )


_validate_costs()
_validate_spells()


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
    unknown name. ``_validate_costs`` makes this unreachable for real
    detector classes, so a warning here means a caller invented a card name
    the model does not emit - most likely a simulator bug.
    """
    key = normalize_name(name)
    cost = CARD_COSTS.get(key)
    if cost is None:
        if key not in _warned_unknown:
            _warned_unknown.add(key)
            logger.warning(
                "Unknown card '%s' (normalized '%s') - defaulting to cost %d. "
                "This name is not one of the detector's card classes; "
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
