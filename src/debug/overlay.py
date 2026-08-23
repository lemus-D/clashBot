"""Debug overlay rendering for the captured frame.

Only used by ``main.py`` when running in ``--debug`` mode.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..game.board import (
    GameBoard,
    ARENA_COLS,
    ARENA_ROWS,
)
from ..game.classes import ARENA_INDEX, IGNORED_ARENA_CLASSES, normalize_name
from ..game.state import GameState

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_COLOR = (0, 255, 0)
TEXT_THICKNESS = 2
TEXT_LINE_HEIGHT = 24

# Troop markers. Friendly/enemy follow the detector's own blue/red naming.
# DROPPED is for a troop the detector found but the observation cannot encode
# - a name missing from ARENA_CLASSES. It should never appear; when it does,
# that troop is absent from the arena tensor.
FRIENDLY_TROOP_COLOR = (255, 160, 0)
ENEMY_TROOP_COLOR = (60, 60, 255)
DROPPED_COLOR = (0, 255, 255)

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


def draw_troops(frame: np.ndarray, board: GameBoard) -> tuple[int, int]:
    """Mark every detected troop on its tile. Returns (encoded, dropped).

    This exists because the arena tensor is the largest part of the
    observation and was, until now, completely invisible: a troop whose name
    the encoder did not recognise was skipped, and the result looked
    identical to an empty arena. A name that cannot be encoded is drawn in
    ``DROPPED_COLOR`` with a ``?``, so "the detector sees it but the
    observation does not" is something you can see rather than infer.
    """
    encoded = dropped = 0
    for y in range(ARENA_ROWS):
        for x in range(ARENA_COLS):
            troop = board.troops_in_arena[y][x]
            if troop is None:
                continue
            key = normalize_name(troop.name)
            if key in IGNORED_ARENA_CLASSES:
                continue
            in_obs = key in ARENA_INDEX
            if in_obs:
                encoded += 1
                color = FRIENDLY_TROOP_COLOR if troop.color == "blue" else ENEMY_TROOP_COLOR
                label = key[:6]
            else:
                dropped += 1
                color = DROPPED_COLOR
                label = f"?{key[:6]}"
            cx = int((x + 0.5) * board.tile_width)
            cy = int((y + 0.5) * board.tile_height)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, label, (cx - 18, cy - 8),
                        TEXT_FONT, 0.35, color, 1)
    return encoded, dropped


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

    Reads ``state.tower_hp`` only - the values the vision layer last pushed
    in - so the overlay never triggers its own vision pass. Values are
    normalized bar fill, shown as percentages; ``DEAD`` marks a tower the
    destruction debounce has committed to.
    """
    lines: list[str] = []
    for label, keys in TOWER_HP_ROWS:
        cells = []
        for name, key in zip(("L", "R", "K"), keys):
            hp = state.tower_hp[key]
            cells.append(f"{name}:{'DEAD' if hp <= 0.0 else f'{hp * 100:3.0f}%'}")
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
    encoded, dropped = draw_troops(out, board)

    lines = [state.get_status_string()]
    if lifecycle_state is not None:
        lines.append(f"Lifecycle: {lifecycle_state}")
    # Whether the arena tensor is actually being populated. "dropped" above 0
    # means the detector and ARENA_CLASSES disagree and those troops are
    # missing from the observation.
    troop_line = f"Troops: {encoded} in obs"
    if dropped:
        troop_line += f"  |  {dropped} DROPPED (not in ARENA_CLASSES)"
    lines.append(troop_line)
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
