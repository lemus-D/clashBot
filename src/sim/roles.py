"""Card ROLES: what job a card does in a deck, derived from its stats.

A deck archetype is a set of roles - a tank to absorb, a mini tank to trade,
a swarm to surround, a building to pull, spells to clear - and both the
structured deck sampler (``env.ArchetypeDeckSpread``) and the punishing
opponents (``opponents.Punisher``) need to ask "which of these is the tank".

The roles are DERIVED FROM THE STAT TABLE, never from a list of card names.
The naming contract in ``game/classes.py`` exists because hand-written card
lists have silently broken this project twice, and a hand-written
``TANKS = ("giant", ...)`` would be the same bug wearing a different hat: it
goes stale the moment the detector gains a card, and it goes stale silently,
because a deck missing its tank still looks like a deck.

Thresholds rather than names, then. They are chosen against the current
12-card pool and are deliberately loose enough to survive it growing:

- ``TANK_HP_MIN`` at 2500 selects Giant (4091) and nothing else today. The
  next-heaviest card is Knight at 1766, so there is a 700-point gap on either
  side of the line - it is not balanced on a knife edge.
- ``MINI_TANK_HP_MIN`` at 1200 selects Knight (1766) and Mini Pekka (1361)
  and excludes Musketeer (720), which is the intended split: the first two
  can lead a small push and survive contact, the third cannot.
- ``SWARM_COUNT_MIN`` at 3 selects Goblins, Spear Goblins and Minions, and
  excludes Archers at 2. Two units is a pair, not a swarm; the distinction
  that matters is whether the card surrounds a tank or merely supports it.

One consequence worth stating: because the pool holds exactly ONE card above
``TANK_HP_MIN``, every structured deck gets Giant. That is a property of a
12-card pool, not of these thresholds, and it stops being true as soon as the
detector learns a second heavy card.
"""

from __future__ import annotations

from ..game.cards import get_card_cost
from .units import SPELL_NAMES, UNIT_STATS, Target

#: A single unit at or above this HP leads a push and absorbs tower fire.
TANK_HP_MIN = 2500.0

#: Below TANK_HP_MIN but at or above this, a card can front a cheap push.
MINI_TANK_HP_MIN = 1200.0

#: Units per card at or above which it counts as a swarm.
SWARM_COUNT_MIN = 3


def is_spell(name: str) -> bool:
    return name in SPELL_NAMES


def is_building(name: str) -> bool:
    stats = UNIT_STATS.get(name)
    return bool(stats and stats.is_building)


def is_troop(name: str) -> bool:
    """A unit that walks. Excludes spells and buildings."""
    return not is_spell(name) and not is_building(name) and name in UNIT_STATS


def is_swarm(name: str) -> bool:
    stats = UNIT_STATS.get(name)
    return bool(stats and not stats.is_building
                and stats.count >= SWARM_COUNT_MIN)


def hits_air(name: str) -> bool:
    """Can this card damage a flying unit?

    Spells count: both of them hit air, and a deck whose only answer to
    Minions is Arrows is a real deck rather than a broken one.
    """
    if is_spell(name):
        return True
    stats = UNIT_STATS.get(name)
    return bool(stats and stats.targets is Target.BOTH)


def is_air_defense(name: str) -> bool:
    """A TROOP that can shoot air, which is not the same as an answer to air.

    Spells hit air too, but a spell is one shot for one elixir payment and
    then it is gone. What a deck needs in the air-defense slot is something
    that keeps shooting - otherwise the second Minion pack of the match goes
    unanswered. So the role deliberately excludes spells even though
    ``hits_air`` includes them.
    """
    return is_troop(name) and hits_air(name)


def hitpoints(name: str) -> float:
    """HP of a SINGLE unit of this card, 0.0 for spells.

    Single rather than total-across-count on purpose: what makes a tank a
    tank is that one body soaks a tower's fire for a long time. Three
    Goblins at 202 each total 606 and tank nothing - they die to one splash
    hit and the push behind them is exposed.
    """
    stats = UNIT_STATS.get(name)
    return float(stats.hp) if stats else 0.0


def is_tank(name: str) -> bool:
    return is_troop(name) and hitpoints(name) >= TANK_HP_MIN


def is_mini_tank(name: str) -> bool:
    return (is_troop(name)
            and MINI_TANK_HP_MIN <= hitpoints(name) < TANK_HP_MIN)


def elixir_value(name: str) -> int:
    return get_card_cost(name)


def unit_elixir_value(name: str) -> float:
    """Elixir value of ONE BODY of this card, not of the card.

    A swarm card puts several units on the field for one payment, so counting
    each body at the full card cost overvalues it by ``count``. Three Goblins
    are worth two elixir between them, not six.

    This distinction is load-bearing and was learned the expensive way. A
    spell-casting bot that counted BODIES fired a 4-elixir Fireball at every
    2-elixir Goblin placement, because three goblins landing on one tile
    always look like a clump of three. Measured, that one mistake cost the
    bot 11 percentage points of win rate against a random policy - it played
    strictly worse than the same bot with spells disabled entirely.
    """
    stats = UNIT_STATS.get(name)
    count = max(1, stats.count) if stats else 1
    return get_card_cost(name) / count


def heaviest(names: list[str]) -> str | None:
    """The best available push leader among ``names``, or None.

    THIS IS THE TANK RULE. It is relative, not absolute: whatever walks and
    has the most HP leads the push, so Giant fronts it when the hand holds
    Giant and Knight fronts it when the hand does not.

    An absolute rule ("wait for a card above TANK_HP_MIN") deadlocks. A deck
    is 8 of 12 cards and a hand is 4 of those 8, so a bot holding out for
    Giant specifically would spend a large fraction of every match holding
    elixir it could not spend - and against the structured decks it would
    still only hold Giant in hand about half the time. Leading with Knight is
    what a human does in that spot.

    Buildings are excluded even when they out-HP the troops: a building does
    not advance, so it cannot lead anything.
    """
    troops = [n for n in names if is_troop(n)]
    if not troops:
        return None
    return max(troops, key=hitpoints)


def describe(name: str) -> str:
    """Role label for debugging and test failure messages."""
    if is_spell(name):
        return "spell"
    if is_building(name):
        return "building"
    if is_tank(name):
        return "tank"
    if is_mini_tank(name):
        return "mini_tank"
    if is_swarm(name):
        return "swarm"
    return "support"
