"""Action space definition and execution.

The action space is intentionally small and discrete so RL policies can
target it directly:

- ``Action(NO_OP)`` - skip this step
- ``Action(hand_index, tile_x, tile_y)`` - place card from hand at tile

``ActionExecutor.execute`` validates: the slot is non-empty, elixir is
sufficient, and the tile is in the playable mask. On success it
performs the mouse drag and tells ``GameState`` to spend the elixir.
Failed actions return ``ActionResult(success=False, reason=...)`` so a
training loop can decide whether to penalize them.

Hand-pixel positions are exposed as fractions of the captured monitor
in ``HAND_CARD_POSITIONS``. CALIBRATE for your BlueStacks crop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pyautogui

from ..game.cards import Card
from ..game.board import GameBoard, ARENA_COLS, ARENA_ROWS, HAND_SIZE
from ..game.state import GameState


# Fractional positions of each hand slot relative to the captured
# monitor. CALIBRATE: open the game, hover the mouse over the centre of
# each card, note the (x_frac, y_frac), and update.
HAND_CARD_POSITIONS: tuple[tuple[float, float], ...] = (
    (0.30, 0.92),
    (0.43, 0.92),
    (0.57, 0.92),
    (0.70, 0.92),
)

DRAG_DURATION_SEC = 0.20
POST_DRAG_PAUSE_SEC = 0.05


# Sentinel for the "do nothing this step" action.
NO_OP_INDEX = -1


@dataclass(frozen=True)
class Action:
    hand_index: int
    tile_x: int = 0
    tile_y: int = 0

    @classmethod
    def no_op(cls) -> "Action":
        return cls(NO_OP_INDEX, 0, 0)

    @property
    def is_no_op(self) -> bool:
        return self.hand_index == NO_OP_INDEX


@dataclass
class ActionResult:
    success: bool
    reason: str = ""
    pixel_target: Optional[tuple[int, int]] = None


def action_space_size() -> int:
    """Total discrete action count: NO_OP + (HAND_SIZE * cols * rows)."""
    return 1 + HAND_SIZE * ARENA_COLS * ARENA_ROWS


def index_to_action(index: int) -> Action:
    if index == 0:
        return Action.no_op()
    index -= 1
    tile_total = ARENA_COLS * ARENA_ROWS
    hand = index // tile_total
    rest = index % tile_total
    tile_y = rest // ARENA_COLS
    tile_x = rest % ARENA_COLS
    return Action(hand_index=hand, tile_x=tile_x, tile_y=tile_y)


def action_to_index(action: Action) -> int:
    if action.is_no_op:
        return 0
    tile_total = ARENA_COLS * ARENA_ROWS
    return 1 + action.hand_index * tile_total + action.tile_y * ARENA_COLS + action.tile_x


class ActionExecutor:
    def __init__(
        self,
        hand_positions: tuple[tuple[float, float], ...] = HAND_CARD_POSITIONS,
        drag_duration: float = DRAG_DURATION_SEC,
    ):
        self.hand_positions = hand_positions
        self.drag_duration = drag_duration

    # ----- helpers -----

    def _hand_pixel(self, hand_index: int, monitor: dict) -> tuple[int, int]:
        fx, fy = self.hand_positions[hand_index]
        return (
            monitor["left"] + int(fx * monitor["width"]),
            monitor["top"] + int(fy * monitor["height"]),
        )

    def _tile_pixel(
        self, board: GameBoard, monitor: dict, tile_x: int, tile_y: int
    ) -> tuple[int, int]:
        local = board.convert_tile_to_image_cord(tile_x, tile_y)
        if local is None:
            raise ValueError(f"Bad tile ({tile_x},{tile_y})")
        return (monitor["left"] + local[0], monitor["top"] + local[1])

    # ----- main entry -----

    def execute(
        self,
        action: Action,
        board: GameBoard,
        state: GameState,
        monitor: dict,
    ) -> ActionResult:
        if action.is_no_op:
            return ActionResult(success=True, reason="no_op")

        if not (0 <= action.hand_index < HAND_SIZE):
            return ActionResult(success=False, reason=f"bad hand_index {action.hand_index}")

        card = board.cards_in_hand[action.hand_index]
        if not isinstance(card, Card):
            return ActionResult(success=False, reason="hand slot empty")

        if not (0 <= action.tile_x < ARENA_COLS and 0 <= action.tile_y < ARENA_ROWS):
            return ActionResult(success=False, reason="tile out of bounds")

        if not board.is_placeable(
            action.tile_x,
            action.tile_y,
            enemy_left_tower_alive=state.is_enemy_left_alive(),
            enemy_right_tower_alive=state.is_enemy_right_alive(),
            enemy_king_active=state.is_enemy_king_active(),
        ):
            return ActionResult(success=False, reason="tile not placeable")

        if state.get_current_elixir() < card.cost:
            return ActionResult(success=False, reason="not enough elixir")

        start = self._hand_pixel(action.hand_index, monitor)
        target = self._tile_pixel(board, monitor, action.tile_x, action.tile_y)

        pyautogui.moveTo(start[0], start[1])
        time.sleep(POST_DRAG_PAUSE_SEC)
        pyautogui.dragTo(target[0], target[1], duration=self.drag_duration, button="left")

        state.spend_elixir(card.cost)
        return ActionResult(success=True, reason="placed", pixel_target=target)
