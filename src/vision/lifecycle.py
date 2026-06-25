"""Visual detection of match lifecycle: menu -> in-match -> postmatch.

Two layered signals are used:

1. Pixel-color sampling at calibrated coordinates (cheap, always on).
2. Optional template matching against PNGs in ``assets/templates/``
   (more reliable when calibrated, silently skipped when missing).

All coordinates and color thresholds are exposed as module-level
constants. They are guesses tuned to a portrait BlueStacks crop and
almost certainly need adjustment for your machine - search for
``CALIBRATE`` to find them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pyautogui


# ----- lifecycle states -----

STATE_MENU = "MENU"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_IN_MATCH = "IN_MATCH"
STATE_POSTMATCH = "POSTMATCH"

# ----- calibration -----

# Sample these (x_frac, y_frac) pixels in the captured frame. CALIBRATE.
ELIXIR_BAR_SAMPLE = (0.50, 0.965)        # purple elixir bar in-match
VICTORY_BANNER_SAMPLE = (0.50, 0.20)     # yellow/gold victory banner area
DEFEAT_BANNER_SAMPLE = (0.50, 0.20)      # blue defeat banner area
COUNTDOWN_SAMPLE = (0.50, 0.50)          # center "3 / 2 / 1" overlay

# Reference colors in BGR; tolerance is per-channel L1 distance.
COLOR_PURPLE_ELIXIR = (180, 60, 200)
COLOR_VICTORY_GOLD = (60, 200, 235)
COLOR_DEFEAT_BLUE = (200, 110, 60)
COLOR_TOLERANCE = 60

# Click targets for auto-rematch (fractions of monitor). CALIBRATE.
OK_BUTTON_FRAC = (0.50, 0.93)
BATTLE_BUTTON_FRAC = (0.50, 0.62)

# Template assets (relative to project root).
# __file__ is src/vision/lifecycle.py, so three dirnames reach the project root.
TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "templates",
)
TEMPLATE_FILES = {
    "battle_button": "battle_button.png",
    "ok_button": "ok_button.png",
    "victory": "victory.png",
    "defeat": "defeat.png",
}
TEMPLATE_MATCH_THRESHOLD = 0.80


@dataclass
class LifecycleSignals:
    state: str
    result: Optional[str] = None
    template_hits: dict[str, float] = None  # type: ignore[assignment]


def _color_distance(a, b) -> float:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2]))


def _sample_pixel(frame: np.ndarray, frac_xy) -> np.ndarray:
    h, w = frame.shape[:2]
    x = int(np.clip(frac_xy[0] * w, 0, w - 1))
    y = int(np.clip(frac_xy[1] * h, 0, h - 1))
    return frame[y, x]


class MatchLifecycle:
    """Detects which lifecycle state the game is in and drives rematches."""

    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = template_dir or TEMPLATE_DIR
        self._templates: dict[str, np.ndarray] = {}
        self._load_templates()
        self._last_state = STATE_MENU

    def _load_templates(self) -> None:
        if not os.path.isdir(self.template_dir):
            return
        for key, fname in TEMPLATE_FILES.items():
            path = os.path.join(self.template_dir, fname)
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    self._templates[key] = img

    def _match_template(self, frame: np.ndarray, key: str) -> float:
        tmpl = self._templates.get(key)
        if tmpl is None or frame is None or frame.size == 0:
            return 0.0
        if tmpl.shape[0] > frame.shape[0] or tmpl.shape[1] > frame.shape[1]:
            return 0.0
        result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    # ----- detection -----

    def detect_state(self, frame: np.ndarray) -> LifecycleSignals:
        hits = {k: self._match_template(frame, k) for k in TEMPLATE_FILES}

        victory_hit = hits.get("victory", 0.0) >= TEMPLATE_MATCH_THRESHOLD
        defeat_hit = hits.get("defeat", 0.0) >= TEMPLATE_MATCH_THRESHOLD
        ok_hit = hits.get("ok_button", 0.0) >= TEMPLATE_MATCH_THRESHOLD
        battle_hit = hits.get("battle_button", 0.0) >= TEMPLATE_MATCH_THRESHOLD

        result: Optional[str] = None
        if victory_hit:
            result = "win"
        elif defeat_hit:
            result = "loss"

        if victory_hit or defeat_hit or ok_hit:
            state = STATE_POSTMATCH
        elif battle_hit:
            state = STATE_MENU
        else:
            state = self._color_based_state(frame)
            if state == STATE_POSTMATCH:
                result = result or self._color_based_result(frame)

        self._last_state = state
        return LifecycleSignals(state=state, result=result, template_hits=hits)

    def _color_based_state(self, frame: np.ndarray) -> str:
        if frame is None or frame.size == 0:
            return self._last_state
        elixir_px = _sample_pixel(frame, ELIXIR_BAR_SAMPLE)
        if _color_distance(elixir_px, COLOR_PURPLE_ELIXIR) <= COLOR_TOLERANCE:
            return STATE_IN_MATCH

        banner_px = _sample_pixel(frame, VICTORY_BANNER_SAMPLE)
        if (
            _color_distance(banner_px, COLOR_VICTORY_GOLD) <= COLOR_TOLERANCE
            or _color_distance(banner_px, COLOR_DEFEAT_BLUE) <= COLOR_TOLERANCE
        ):
            return STATE_POSTMATCH

        # Bias toward keeping the previous state instead of bouncing to MENU
        # on a single noisy frame.
        if self._last_state == STATE_IN_MATCH:
            return STATE_IN_MATCH
        return STATE_MENU

    def _color_based_result(self, frame: np.ndarray) -> Optional[str]:
        banner_px = _sample_pixel(frame, VICTORY_BANNER_SAMPLE)
        if _color_distance(banner_px, COLOR_VICTORY_GOLD) <= COLOR_TOLERANCE:
            return "win"
        if _color_distance(banner_px, COLOR_DEFEAT_BLUE) <= COLOR_TOLERANCE:
            return "loss"
        return None

    # ----- side effects -----

    def auto_rematch(self, monitor: dict, between_clicks_sec: float = 1.5) -> None:
        """Click ``OK`` then ``Battle`` to start a new match.

        Coordinates are taken from ``OK_BUTTON_FRAC`` and
        ``BATTLE_BUTTON_FRAC``. The caller passes the ``mss`` monitor
        dict so we can convert to global screen coordinates.
        """
        import time

        ok_x = monitor["left"] + int(OK_BUTTON_FRAC[0] * monitor["width"])
        ok_y = monitor["top"] + int(OK_BUTTON_FRAC[1] * monitor["height"])
        pyautogui.moveTo(ok_x, ok_y)
        pyautogui.click()
        time.sleep(between_clicks_sec)

        battle_x = monitor["left"] + int(BATTLE_BUTTON_FRAC[0] * monitor["width"])
        battle_y = monitor["top"] + int(BATTLE_BUTTON_FRAC[1] * monitor["height"])
        pyautogui.moveTo(battle_x, battle_y)
        pyautogui.click()
