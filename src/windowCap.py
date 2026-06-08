"""Debug overlay rendering for the captured frame.

Capture, mouse control, and the main loop now live in
``capture.ScreenCapture``, ``actions.ActionExecutor``, and
``environment.ClashEnv``. This module only owns visualization helpers
used by ``main.py`` when running in ``--debug`` mode.
"""

from __future__ import annotations

import cv2
import numpy as np

from .gameBoard import GameBoard, ARENA_COLS, ARENA_ROWS


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


def render_debug_overlay(
    frame: np.ndarray,
    board: GameBoard,
    state,
    detection_summary: dict | None = None,
    lifecycle_state: str | None = None,
) -> np.ndarray:
    """Annotate a captured frame with grid + status text.

    Detector boxes/labels are intentionally NOT drawn here so this is
    cheap to call without a Supervision dependency. ``main.py`` mixes
    in the annotated detection frame separately when desired.
    """
    out = frame.copy()
    out = draw_tile_grid(out, board)

    lines: list[str] = []
    if state is not None:
        try:
            lines.append(state.get_status_string())
        except Exception:
            pass
    if lifecycle_state is not None:
        lines.append(f"Lifecycle: {lifecycle_state}")
    if detection_summary is not None:
        lines.append(f"Cards: {len(detection_summary.get('cards_in_hand', []))}")
        lines.append(f"Troops: {len(detection_summary.get('troops_on_board', []))}")

    y_off = 30
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y_off),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        y_off += 24

    return out
