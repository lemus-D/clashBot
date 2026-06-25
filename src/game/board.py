"""Logical board state: a 4-card hand and a 9x16 arena grid.

The arena grid is the standard Clash Royale tile resolution (9 columns x
16 rows). The bridge sits between rows 7 and 8, so friendly placement is
restricted to ``y >= 8`` unless a tower has been destroyed (then the
opposing top quadrant becomes placeable).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .cards import Card, Troop, BlankSpace, EMPTY_CARD, EMPTY_TILE, is_empty


ARENA_COLS = 9
ARENA_ROWS = 16
HAND_SIZE = 4

FRIENDLY_HALF_START_ROW = 8

TROOP_CLASSES: tuple[str, ...] = (
    "knight",
    "archers",
    "minions",
    "musketeer",
    "valkyrie",
    "hogrider",
    "babydragon",
    "minipekka",
    "wizard",
    "icewizard",
    "bomber",
    "skeletons",
    "skeletonarmy",
    "goblins",
    "spear goblins",
    "barbarians",
    "minionhorde",
    "balloon",
    "prince",
    "darkprince",
    "witch",
    "nightwitch",
    "giant",
    "lavahound",
    "golem",
    "pekka",
    "megaknight",
    "electrowizard",
    "infernodragon",
    "magicarcher",
    "bandit",
    "executioner",
    "bowler",
    "hunter",
    "battleram",
    "miner",
    "princess",
    "guards",
    "cannon",
    "tesla",
    "infernotower",
    "bombtower",
    "tombstone",
    "mortar",
    "xbow",
    "elixircollector",
    "icegolem",
    "bats",
    "icespirit",
    "firespirit",
    "electrospirit",
)


def _normalize_troop_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


_TROOP_INDEX: dict[str, int] = {
    _normalize_troop_name(n): i for i, n in enumerate(TROOP_CLASSES)
}


class GameBoard:
    def __init__(self, monitor_width: int, monitor_height: int):
        self.monitor_width = monitor_width
        self.monitor_height = monitor_height

        self.tile_width = monitor_width / ARENA_COLS
        self.tile_height = monitor_height / ARENA_ROWS

        self.cards_in_hand: list = [EMPTY_CARD] * HAND_SIZE
        self.troops_in_arena: list[list] = [
            [EMPTY_TILE for _ in range(ARENA_COLS)] for _ in range(ARENA_ROWS)
        ]

    # ----- mutation helpers -----

    def add_card_to_hand(self, card: Card, position: int) -> None:
        if 0 <= position < HAND_SIZE:
            self.cards_in_hand[position] = card
        else:
            print(f"Invalid hand position: {position}. Must be 0-{HAND_SIZE-1}.")

    def add_troop_to_arena(self, troop: Troop, x_cord: int, y_cord: int) -> None:
        if 0 <= x_cord < ARENA_COLS and 0 <= y_cord < ARENA_ROWS:
            troop.tile_x = x_cord
            troop.tile_y = y_cord
            self.troops_in_arena[y_cord][x_cord] = troop
        else:
            print(f"Invalid arena position: ({x_cord}, {y_cord})")

    def clear_arena(self) -> None:
        for row in self.troops_in_arena:
            for x in range(ARENA_COLS):
                row[x] = EMPTY_TILE

    def clear_hand(self) -> None:
        for i in range(HAND_SIZE):
            self.cards_in_hand[i] = EMPTY_CARD

    # ----- coordinate conversions -----

    def convert_image_cord_to_tile(self, x_image_cord: float, y_image_cord: float):
        if x_image_cord < 0 or x_image_cord > self.monitor_width:
            return None
        if y_image_cord < 0 or y_image_cord > self.monitor_height:
            return None
        tile_x = int(x_image_cord / self.tile_width)
        tile_y = int(y_image_cord / self.tile_height)
        tile_x = min(tile_x, ARENA_COLS - 1)
        tile_y = min(tile_y, ARENA_ROWS - 1)
        return (tile_x, tile_y)

    def convert_tile_to_image_cord(self, tile_x: int, tile_y: int):
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
    ) -> bool:
        """Whether the friendly side may place a regular ground troop here.

        Default rule: rows 8-15 (friendly half). When an enemy princess
        tower falls, the corresponding top quadrant unlocks. Activating the
        enemy king tower unlocks the full enemy half. The river rows (7
        and 8) remain non-placeable for ground units in standard play.
        Spells and tornado-style cards ignore this mask; callers that need
        spell-specific rules should override.
        """
        if not (0 <= tile_x < ARENA_COLS and 0 <= tile_y < ARENA_ROWS):
            return False

        # Bridge tiles are not placeable for ground units.
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
    ) -> np.ndarray:
        mask = np.zeros((ARENA_ROWS, ARENA_COLS), dtype=np.uint8)
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                if self.is_placeable(
                    x,
                    y,
                    enemy_left_tower_alive,
                    enemy_right_tower_alive,
                    enemy_king_active,
                ):
                    mask[y, x] = 1
        return mask

    # ----- detection consumption -----

    def process_detections(self, detections) -> dict:
        cards_detected: list[dict] = []
        cards_filtered: list[dict] = []
        troops_detected: list[dict] = []

        for i in range(len(detections)):
            class_name = detections.data["class_name"][i]
            bbox = detections.xyxy[i]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            bbox_area = bbox_w * bbox_h

            if class_name.startswith("card"):
                card_name = class_name[4:]
                cards_detected.append(
                    {
                        "name": card_name,
                        "pixel_coords": (center_x, center_y),
                        "bbox": bbox,
                        "width": bbox_w,
                        "height": bbox_h,
                        "area": bbox_area,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                )
            elif class_name.startswith("blue") or class_name.startswith("red"):
                if class_name.startswith("blue"):
                    color = "blue"
                    troop_name = class_name[4:]
                else:
                    color = "red"
                    troop_name = class_name[3:]
                tile_coords = self.convert_image_cord_to_tile(center_x, center_y)
                if tile_coords:
                    troops_detected.append(
                        {
                            "name": troop_name,
                            "color": color,
                            "pixel_coords": (center_x, center_y),
                            "tile_coords": tile_coords,
                        }
                    )
                    self.add_troop_to_arena(
                        Troop(troop_name, color), tile_coords[0], tile_coords[1]
                    )

        if cards_detected:
            cards_in_hand = self.filter_cards_in_hand(cards_detected)
            cards_in_hand.sort(key=lambda c: c["center_x"])
            for position, info in enumerate(cards_in_hand[:HAND_SIZE]):
                self.add_card_to_hand(Card(info["name"]), position)
            cards_filtered = [c for c in cards_detected if c not in cards_in_hand]
        else:
            cards_in_hand = []

        return {
            "cards_in_hand": cards_in_hand,
            "cards_filtered": cards_filtered,
            "troops_on_board": troops_detected,
        }

    def filter_cards_in_hand(self, cards_detected: list[dict]) -> list[dict]:
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

    # ----- ML / debug serialization -----

    def to_tensor(self) -> np.ndarray:
        """One-hot encode the arena as ``(ARENA_ROWS, ARENA_COLS, channels)``.

        Channels = ``len(TROOP_CLASSES) * 2`` (blue/friendly first half,
        red/enemy second half). Unknown troop names are silently ignored
        rather than crashing - they show up as all-zero tile vectors.
        """
        n_classes = len(TROOP_CLASSES)
        channels = n_classes * 2
        tensor = np.zeros((ARENA_ROWS, ARENA_COLS, channels), dtype=np.float32)
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                cell = self.troops_in_arena[y][x]
                if not isinstance(cell, Troop):
                    continue
                key = _normalize_troop_name(cell.name)
                idx = _TROOP_INDEX.get(key)
                if idx is None:
                    continue
                offset = 0 if cell.color == "blue" else n_classes
                tensor[y, x, offset + idx] = 1.0
        return tensor

    def hand_to_tensor(self, card_classes: Optional[Iterable[str]] = None) -> np.ndarray:
        """One-hot encode the hand as ``(HAND_SIZE, num_cards)``.

        Defaults to ``TROOP_CLASSES`` for the vocabulary; pass an explicit
        list if you want a different card vocabulary (e.g. include spells
        that aren't in ``TROOP_CLASSES``).
        """
        vocab = list(card_classes) if card_classes is not None else list(TROOP_CLASSES)
        index = {_normalize_troop_name(n): i for i, n in enumerate(vocab)}
        out = np.zeros((HAND_SIZE, len(vocab)), dtype=np.float32)
        for slot, card in enumerate(self.cards_in_hand):
            if not isinstance(card, Card):
                continue
            key = _normalize_troop_name(card.name)
            idx = index.get(key)
            if idx is not None:
                out[slot, idx] = 1.0
        return out

    def hand_costs(self) -> np.ndarray:
        out = np.zeros((HAND_SIZE,), dtype=np.float32)
        for i, card in enumerate(self.cards_in_hand):
            if isinstance(card, Card):
                out[i] = float(card.cost)
        return out

    def get_board_state(self) -> str:
        state = "=== CARDS IN HAND (Position 0-3, Left to Right) ===\n"
        for i, card in enumerate(self.cards_in_hand):
            if isinstance(card, Card):
                state += f"Position {i}: {card.name} (Cost: {card.cost})\n"
            else:
                state += f"Position {i}: Empty\n"

        state += "\n=== TROOPS IN ARENA (9x16 grid) ===\n"
        state += "   " + "".join(f"{i:3}" for i in range(ARENA_COLS)) + "\n"
        for y in range(ARENA_ROWS):
            state += f"{y:2} "
            for x in range(ARENA_COLS):
                cell = self.troops_in_arena[y][x]
                if isinstance(cell, Troop):
                    state += f" {cell.color[0].upper()}{cell.name[0]} "
                else:
                    state += " . "
            state += "\n"
        return state
