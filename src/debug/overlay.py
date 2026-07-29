"""Debug overlay rendering for the captured frame.

Only used by ``main.py`` when running in ``--debug`` mode.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..game.board import GameBoard, ARENA_COLS, ARENA_ROWS
from ..game.state import GameState

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_COLOR = (0, 255, 0)
TEXT_THICKNESS = 2
TEXT_LINE_HEIGHT = 24

# Perceived tower HP, top row first so it reads in screen order (enemy
# towers are at the top of the frame).
TOWER_HP_ROWS: tuple[tuple[str, tuple[str, str, str]], ...] = (
    ("Enemy towers", ("enemy_left", "enemy_right", "enemy_king")),
    ("Friendly towers", ("friendly_left", "friendly_right", "friendly_king")),
)
UNKNOWN_HP = "--"


def draw_tile_grid(frame: np.ndarray, game_board: GameBoard) -> np.ndarray:
    color = (0, 255, 255)
    thickness = 1

    for x in range(ARENA_COLS + 1):
        x_pos = int(x * game_board.tile_width)
        cv2.line(frame, (x_pos, 0), (x_pos, frame.shape[0]), color, thickness)
    for y in range(ARENA_ROWS + 1):
        y_pos = int(y * game_board.tile_height)
        cv2.line(frame, (0, y_pos), (frame.shape[1], y_pos), color, thickness)

    for x in range(ARENA_COLS):
        for y in range(ARENA_ROWS):
            if (x + y) % 2 == 0:
                label = f"{x},{y}"
                cx = int((x + 0.5) * game_board.tile_width)
                cy = int((y + 0.5) * game_board.tile_height)
                cv2.putText(
                    frame,
                    label,
                    (cx - 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )
    return frame


def draw_text_lines(frame: np.ndarray, lines: list[str], top_y: int) -> None:
    """Draw a stack of status lines in place, first baseline at ``top_y``."""
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, top_y + i * TEXT_LINE_HEIGHT),
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
        )


def tower_hp_lines(state: GameState) -> list[str]:
    """One line per side of the tower HP the program currently believes.

    Reads ``state.tower_hp`` only - the values the OCR reader last pushed
    in - so the overlay never triggers its own vision pass. A tower whose
    HP was never successfully read shows ``--`` instead of a number.
    """
    lines: list[str] = []
    for label, keys in TOWER_HP_ROWS:
        cells = []
        for name, key in zip(("L", "R", "K"), keys):
            hp = state.tower_hp[key]
            cells.append(f"{name}:{UNKNOWN_HP if hp is None else hp}")
        lines.append(f"{label}  " + "  ".join(cells))
    return lines


def render_debug_overlay(
    frame: np.ndarray,
    board: GameBoard,
    state: GameState,
    lifecycle_state: str | None = None,
) -> np.ndarray:
    """Annotate a captured frame with the tile grid and status text."""
    out = frame.copy()
    out = draw_tile_grid(out, board)

    lines = [state.get_status_string()]
    if lifecycle_state is not None:
        lines.append(f"Lifecycle: {lifecycle_state}")
    draw_text_lines(out, lines, top_y=30)

    # Bottom-left, above the frame edge: clear of the status stack at the
    # top and of the arena rows, which end above the hand-card strip.
    hp_lines = tower_hp_lines(state)
    draw_text_lines(
        out,
        hp_lines,
        top_y=out.shape[0] - 12 - TEXT_LINE_HEIGHT * (len(hp_lines) - 1),
    )

    return out
