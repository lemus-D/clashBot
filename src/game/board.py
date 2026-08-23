"""Logical board state: a 4-card hand and a 9x16 arena grid.

The arena grid is the standard Clash Royale tile resolution (9 columns x
16 rows). The bridge sits between rows 7 and 8, so friendly placement is
restricted to ``y >= 8`` unless a tower has been destroyed (then the
opposing top quadrant becomes placeable).

Empty hand slots and arena tiles are ``None``.
"""

from __future__ import annotations

import logging

import numpy as np

from .cards import Card, Troop, is_spell
from .classes import (
    ARENA_CLASSES,
    ARENA_INDEX,
    CARD_CLASSES,
    CARD_INDEX,
    IGNORED_ARENA_CLASSES,
    normalize_name,
)

logger = logging.getLogger(__name__)

ARENA_COLS = 9
ARENA_ROWS = 16
HAND_SIZE = 4

FRIENDLY_HALF_START_ROW = 8

# The arena and the hand use SEPARATE class lists, both generated from the
# detector - see ``classes.py`` for why they are not one list. Briefly:
# Goblin Brawler is an arena unit with no card, so a shared list gives the
# hand a channel that can never fire, and every spawn-only unit added later
# would do the same.

# Unencodable troop names already reported, so the warning fires once per
# distinct name instead of once per tile per frame.
_warned_unknown_troops: set[str] = set()


class GameBoard:
    def __init__(self, monitor_width: int, monitor_height: int):
        self.monitor_width = monitor_width
        self.monitor_height = monitor_height

        self.tile_width = monitor_width / ARENA_COLS
        self.tile_height = monitor_height / ARENA_ROWS

        self.cards_in_hand: list[Card | None] = [None] * HAND_SIZE
        self.troops_in_arena: list[list[Troop | None]] = [
            [None] * ARENA_COLS for _ in range(ARENA_ROWS)
        ]

    # ----- mutation helpers -----

    def clear_arena(self) -> None:
        self.troops_in_arena = [[None] * ARENA_COLS for _ in range(ARENA_ROWS)]

    def clear_hand(self) -> None:
        self.cards_in_hand = [None] * HAND_SIZE

    # ----- coordinate conversions -----

    def convert_image_cord_to_tile(
        self, x_image_cord: float, y_image_cord: float
    ) -> tuple[int, int] | None:
        if not (0 <= x_image_cord <= self.monitor_width):
            return None
        if not (0 <= y_image_cord <= self.monitor_height):
            return None
        tile_x = min(int(x_image_cord / self.tile_width), ARENA_COLS - 1)
        tile_y = min(int(y_image_cord / self.tile_height), ARENA_ROWS - 1)
        return (tile_x, tile_y)

    def convert_tile_to_image_cord(
        self, tile_x: int, tile_y: int
    ) -> tuple[int, int] | None:
        if not (0 <= tile_x < ARENA_COLS and 0 <= tile_y < ARENA_ROWS):
            return None
        pixel_x = int((tile_x + 0.5) * self.tile_width)
        pixel_y = int((tile_y + 0.5) * self.tile_height)
        return (pixel_x, pixel_y)

    # ----- placement rules -----

    def is_placeable(
        self,
        tile_x: int,
        tile_y: int,
        enemy_left_tower_alive: bool = True,
        enemy_right_tower_alive: bool = True,
        enemy_king_active: bool = False,
        spell: bool = False,
    ) -> bool:
        """Whether the friendly side may place this card here.

        Troop rule: rows 8-15 (friendly half). When an enemy princess
        tower falls, the corresponding top quadrant unlocks. Activating
        the enemy king tower unlocks the full enemy half. Bridge row 7 is
        never placeable for ground units.

        SPELL rule: anywhere in the arena. A spell whose only legal targets
        were on your own half could never hit an enemy tower, which made
        Fireball and Arrows strictly dead cards.
        """
        if not (0 <= tile_x < ARENA_COLS and 0 <= tile_y < ARENA_ROWS):
            return False
        if spell:
            return True
        if tile_y == 7:
            return False
        if tile_y >= FRIENDLY_HALF_START_ROW:
            return True
        if enemy_king_active:
            return True

        midline = ARENA_COLS // 2
        if tile_x < midline and not enemy_left_tower_alive:
            return True
        if tile_x > midline and not enemy_right_tower_alive:
            return True
        return False

    def get_placeable_mask(
        self,
        enemy_left_tower_alive: bool = True,
        enemy_right_tower_alive: bool = True,
        enemy_king_active: bool = False,
        spell: bool = False,
    ) -> np.ndarray:
        """Tile mask for TROOPS by default; ``spell=True`` gives the whole
        arena. The observation carries the troop mask plus a per-slot
        ``hand_is_spell`` flag rather than four full masks - the only
        card-dependent rule is spell-vs-troop, so 4 floats say everything
        that 4x144 would."""
        mask = np.zeros((ARENA_ROWS, ARENA_COLS), dtype=np.uint8)
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                if self.is_placeable(
                    x,
                    y,
                    enemy_left_tower_alive,
                    enemy_right_tower_alive,
                    enemy_king_active,
                    spell=spell,
                ):
                    mask[y, x] = 1
        return mask

    # ----- detection consumption -----

    def process_detections(self, detections) -> None:
        """Populate the hand and arena from a Supervision detections object.

        Class-name convention: ``card<Name>`` for hand cards,
        ``blue<Name>`` / ``red<Name>`` for troops on the board.
        """
        cards_detected: list[dict] = []

        for i in range(len(detections)):
            class_name = detections.data["class_name"][i]
            x1, y1, x2, y2 = detections.xyxy[i]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if class_name.startswith("card"):
                cards_detected.append(
                    {
                        "name": class_name[4:],
                        "area": (x2 - x1) * (y2 - y1),
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                )
            elif class_name.startswith(("blue", "red")):
                color = "blue" if class_name.startswith("blue") else "red"
                troop_name = class_name[len(color):]
                tile = self.convert_image_cord_to_tile(center_x, center_y)
                if tile:
                    tile_x, tile_y = tile
                    self.troops_in_arena[tile_y][tile_x] = Troop(
                        troop_name, color, tile_x, tile_y
                    )

        if cards_detected:
            hand = self.filter_cards_in_hand(cards_detected)
            hand.sort(key=lambda c: c["center_x"])
            for position, info in enumerate(hand[:HAND_SIZE]):
                self.cards_in_hand[position] = Card(info["name"])

    def filter_cards_in_hand(self, cards_detected: list[dict]) -> list[dict]:
        """Keep the (up to) HAND_SIZE detections that look like hand cards:
        near-median size and vertical position, not hugging the left edge."""
        if len(cards_detected) <= HAND_SIZE:
            return cards_detected

        areas = sorted(c["area"] for c in cards_detected)
        median_area = areas[len(areas) // 2]
        ys = sorted(c["center_y"] for c in cards_detected)
        median_y = ys[len(ys) // 2]

        size_threshold = 0.6
        y_threshold = self.monitor_height * 0.1
        left_edge_threshold = self.monitor_width * 0.15

        valid = [
            c
            for c in cards_detected
            if c["area"] >= median_area * size_threshold
            and abs(c["center_y"] - median_y) < y_threshold
            and c["center_x"] > left_edge_threshold
        ]
        if len(valid) > HAND_SIZE:
            valid.sort(key=lambda c: c["area"], reverse=True)
            valid = valid[:HAND_SIZE]
        return valid

    # ----- ML serialization -----

    def to_tensor(self) -> np.ndarray:
        """One-hot encode the arena as ``(ARENA_ROWS, ARENA_COLS, channels)``.

        Channels = ``len(ARENA_CLASSES) * 2`` (blue/friendly first half,
        red/enemy second half).

        A troop whose name is not in ``ARENA_CLASSES`` cannot be encoded, so
        it is omitted - but it is WARNED about once per distinct name, not
        dropped quietly. Silence here hid a real bug: the detector emits
        singular names ("minion") while the hand-written list held plurals
        ("minions"), so four of the most common units never appeared in any
        observation while the arena - the large majority of the flattened
        vector - looked merely empty. Classes in ``IGNORED_ARENA_CLASSES``
        are excluded from the warning because omitting them is a decision.
        """
        n_classes = len(ARENA_CLASSES)
        tensor = np.zeros((ARENA_ROWS, ARENA_COLS, n_classes * 2), dtype=np.float32)
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                troop = self.troops_in_arena[y][x]
                if troop is None:
                    continue
                key = normalize_name(troop.name)
                idx = ARENA_INDEX.get(key)
                if idx is None:
                    self._warn_unknown_troop(troop.name, key)
                    continue
                offset = 0 if troop.color == "blue" else n_classes
                tensor[y, x, offset + idx] = 1.0
        return tensor

    @staticmethod
    def _warn_unknown_troop(name: str, key: str) -> None:
        """Warn once per distinct unencodable troop name."""
        if key in IGNORED_ARENA_CLASSES or key in _warned_unknown_troops:
            return
        _warned_unknown_troops.add(key)
        logger.warning(
            "Troop '%s' (normalized '%s') is not in ARENA_CLASSES - it is "
            "being LEFT OUT of the arena observation entirely. The class "
            "manifest and the detector disagree; regenerate with "
            "`python -m src.main --derive-classes`.",
            name,
            key,
        )

    def hand_to_tensor(self) -> np.ndarray:
        """One-hot encode the hand as ``(HAND_SIZE, len(CARD_CLASSES))``.

        Keyed on CARD_CLASSES, not the arena list: a card that cannot be
        held (Goblin Brawler, and every spawn-only unit after it) has no
        slot here at all.
        """
        out = np.zeros((HAND_SIZE, len(CARD_CLASSES)), dtype=np.float32)
        for slot, card in enumerate(self.cards_in_hand):
            if card is None:
                continue
            idx = CARD_INDEX.get(normalize_name(card.name))
            if idx is not None:
                out[slot, idx] = 1.0
        return out

    def hand_is_spell(self) -> np.ndarray:
        """1.0 for each hand slot holding a spell, else 0.0.

        This is what tells a policy that ``playable_mask`` does not apply to
        that slot - a spell may be cast on any tile.
        """
        out = np.zeros((HAND_SIZE,), dtype=np.float32)
        for i, card in enumerate(self.cards_in_hand):
            if card is not None and is_spell(card.name):
                out[i] = 1.0
        return out

    def hand_costs(self) -> np.ndarray:
        out = np.zeros((HAND_SIZE,), dtype=np.float32)
        for i, card in enumerate(self.cards_in_hand):
            if card is not None:
                out[i] = float(card.cost)
        return out
