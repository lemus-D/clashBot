"""Lightweight data classes for cards, troops, and empty placeholders.

``EMPTY_CARD`` and ``EMPTY_TILE`` are module-level singletons used by
``GameBoard`` to fill empty hand slots and arena tiles. Reusing one object
per kind keeps per-frame allocation flat instead of allocating
``9 * 16 = 144`` ``BlankSpace`` instances every clear.
"""

from __future__ import annotations

from typing import Optional

from .cardDatabase import get_card_cost


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
