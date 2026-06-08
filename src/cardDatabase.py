"""Static lookup of Clash Royale card elixir costs.

The Roboflow detector emits class names like "cardknight" or "cardfireball".
After the "card" prefix is stripped in GameBoard.process_detections, the
remaining suffix is the dictionary key here.

Names are stored lowercase with no spaces or punctuation so callers can
normalize input before lookup.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

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
