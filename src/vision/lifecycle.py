"""Visual detection of match lifecycle: menu -> in-match -> postmatch.

Two layered signals:

1. A magenta-pixel fraction inside the elixir bar means in-match. This
   is checked first and short-circuits: it costs microseconds, whereas
   the template sweep below is four full-frame passes.
2. Template matching against PNGs in ``src/assets/templates/``
   (battle button, OK button, victory/defeat banners), for the frames
   where the elixir bar is absent. Banner colors are the fallback when
   no template hits.

Per-machine coordinates and thresholds are module-level constants
marked ``CALIBRATE``.

Every verdict also carries the raw numbers it was decided from (magenta
fraction, banner pixel and its colour distances, template scores) and a
``path`` naming which of the layers above produced it. Those are pure
diagnostics — nothing reads them to decide anything — and exist because
a wrong verdict is otherwise indistinguishable from a right one after
the fact. ``ClashEnv`` writes them as ``{"type": "diag"}`` records while
recording.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyautogui


# ----- lifecycle states -----

STATE_MENU = "MENU"
STATE_IN_MATCH = "IN_MATCH"
STATE_POSTMATCH = "POSTMATCH"

# ----- calibration -----

# (x_frac, y_frac) pixel sampled for the postmatch banner color. CALIBRATE.
VICTORY_BANNER_SAMPLE = (0.50, 0.20)

# In-match detection: fraction of magenta pixels inside the left part of
# the elixir bar. A patch is far more robust than a single pixel - the
# bar is thin and crossed by white segment ticks. (x0, y0, x1, y1)
# fractions of the frame. CALIBRATE.
ELIXIR_BAR_PATCH = (0.20, 0.955, 0.40, 0.978)
ELIXIR_MAGENTA_MIN_FRAC = 0.10

# Reference colors in BGR; tolerance is per-channel L1 distance.
COLOR_VICTORY_GOLD = (60, 200, 235)
COLOR_DEFEAT_BLUE = (200, 110, 60)
COLOR_TOLERANCE = 60

# Fallback click targets when template matching fails (fractions of
# monitor). CALIBRATE.
OK_BUTTON_FRAC = (0.50, 0.93)
BATTLE_BUTTON_FRAC = (0.50, 0.78)

# Templates were captured at a larger window size than the current
# capture; resize them at load so matchTemplate scores stay high.
# CALIBRATE when the BlueStacks window size changes.
TEMPLATE_SCALE = 0.74

# Template assets live next to the source tree in src/assets/templates.
# __file__ is src/vision/lifecycle.py, so two dirnames reach src/.
TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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

# ----- verdict paths -----
#
# Which layer actually decided a frame's state. Recorded so a wrong
# verdict can be traced to the signal that produced it instead of being
# guessed at from the state alone.
PATH_ELIXIR_GATE = "elixir_gate"        # magenta fraction cleared the gate
PATH_TEMPLATE_SWEEP = "template_sweep"  # a template scored over threshold
PATH_BANNER_COLOR = "banner_color"      # colour fallback matched a banner
PATH_HYSTERESIS = "hysteresis"          # no signal; previous state kept
PATH_DEFAULT_MENU = "default_menu"      # no signal and no state to keep


def _color_distance(a, b) -> float:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) + abs(int(a[2]) - int(b[2]))


@dataclass(frozen=True)
class BannerSample:
    """The postmatch banner pixel and its distance to each reference colour.

    Both the state fallback and the win/loss readout are decided purely
    from these two distances, so sampling once and passing this around
    replaces two independent re-samples of the same pixel and makes the
    numbers recordable.
    """

    bgr: tuple[int, int, int]
    dist_victory: float
    dist_defeat: float

    @classmethod
    def at(cls, frame: np.ndarray, frac_xy: tuple[float, float]) -> "BannerSample":
        px = _sample_pixel(frame, frac_xy)
        return cls(
            bgr=(int(px[0]), int(px[1]), int(px[2])),
            dist_victory=_color_distance(px, COLOR_VICTORY_GOLD),
            dist_defeat=_color_distance(px, COLOR_DEFEAT_BLUE),
        )

    def matches_a_banner(self) -> bool:
        return (
            self.dist_victory <= COLOR_TOLERANCE
            or self.dist_defeat <= COLOR_TOLERANCE
        )

    def result(self) -> str | None:
        if self.dist_victory <= COLOR_TOLERANCE:
            return "win"
        if self.dist_defeat <= COLOR_TOLERANCE:
            return "loss"
        return None


@dataclass
class LifecycleSignals:
    state: str
    result: str | None = None
    template_hits: dict[str, float] = field(default_factory=dict)

    # ----- diagnostics -----
    # Raw decision inputs, recorded but never acted on. Deliberately here
    # and not in the observation dict: the observation schema is hashed
    # into recordings and trained checkpoints, so it must not grow to
    # carry debug data.
    path: str = PATH_HYSTERESIS
    elixir_magenta_frac: float = 0.0
    elixir_bar_visible: bool = False
    banner: BannerSample | None = None


def _sample_pixel(frame: np.ndarray, frac_xy) -> np.ndarray:
    h, w = frame.shape[:2]
    x = int(np.clip(frac_xy[0] * w, 0, w - 1))
    y = int(np.clip(frac_xy[1] * h, 0, h - 1))
    return frame[y, x]


class MatchLifecycle:
    """Detects which lifecycle state the game is in and drives rematches."""

    def __init__(self, template_dir: str | None = None):
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
                    if TEMPLATE_SCALE != 1.0:
                        img = cv2.resize(
                            img, None, fx=TEMPLATE_SCALE, fy=TEMPLATE_SCALE
                        )
                    self._templates[key] = img

    def _locate_template(
        self, frame: np.ndarray, key: str
    ) -> tuple[float, tuple[float, float] | None]:
        """Best match score and its centre as (x_frac, y_frac), or None."""
        tmpl = self._templates.get(key)
        if tmpl is None or frame is None or frame.size == 0:
            return 0.0, None
        if tmpl.shape[0] > frame.shape[0] or tmpl.shape[1] > frame.shape[1]:
            return 0.0, None
        result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        h, w = frame.shape[:2]
        cx = (max_loc[0] + tmpl.shape[1] / 2) / w
        cy = (max_loc[1] + tmpl.shape[0] / 2) / h
        return float(max_val), (cx, cy)

    # ----- detection -----

    def detect_state(self, frame: np.ndarray) -> LifecycleSignals:
        # Cheap gate first: the elixir bar is on screen only while a match is
        # running, and none of the four templates (battle / ok / victory /
        # defeat) can be on screen at the same time as it. So a frame with the
        # bar visible is IN_MATCH regardless of what matchTemplate would say,
        # and the four full-frame sweeps can be skipped. The frame a match ends
        # on loses the bar, so the postmatch transition is still seen at once.
        bar_visible = False
        magenta_frac = 0.0
        if frame is not None and frame.size:
            bar_visible, magenta_frac = self._elixir_bar_visible(frame)
        if bar_visible:
            self._last_state = STATE_IN_MATCH
            return LifecycleSignals(
                state=STATE_IN_MATCH,
                path=PATH_ELIXIR_GATE,
                elixir_magenta_frac=magenta_frac,
                elixir_bar_visible=True,
            )

        hits = {k: self._locate_template(frame, k)[0] for k in TEMPLATE_FILES}

        victory_hit = hits["victory"] >= TEMPLATE_MATCH_THRESHOLD
        defeat_hit = hits["defeat"] >= TEMPLATE_MATCH_THRESHOLD
        ok_hit = hits["ok_button"] >= TEMPLATE_MATCH_THRESHOLD
        battle_hit = hits["battle_button"] >= TEMPLATE_MATCH_THRESHOLD

        result: str | None = None
        if victory_hit:
            result = "win"
        elif defeat_hit:
            result = "loss"

        banner: BannerSample | None = None
        if victory_hit or defeat_hit or ok_hit:
            state = STATE_POSTMATCH
            path = PATH_TEMPLATE_SWEEP
        elif battle_hit:
            state = STATE_MENU
            path = PATH_TEMPLATE_SWEEP
        else:
            state, path, banner = self._color_based_state(frame)
            if state == STATE_POSTMATCH and banner is not None:
                result = result or banner.result()

        self._last_state = state
        return LifecycleSignals(
            state=state,
            result=result,
            template_hits=hits,
            path=path,
            elixir_magenta_frac=magenta_frac,
            elixir_bar_visible=bar_visible,
            banner=banner,
        )

    def _color_based_state(
        self, frame: np.ndarray
    ) -> tuple[str, str, BannerSample | None]:
        """State from banner colour alone, plus the verdict path and the
        banner sample it was read from (None when no pixel was sampled)."""
        if frame is None or frame.size == 0:
            return self._last_state, PATH_HYSTERESIS, None
        if self._elixir_bar_visible(frame)[0]:
            return STATE_IN_MATCH, PATH_ELIXIR_GATE, None

        banner = BannerSample.at(frame, VICTORY_BANNER_SAMPLE)
        if banner.matches_a_banner():
            return STATE_POSTMATCH, PATH_BANNER_COLOR, banner

        # Bias toward keeping the previous state instead of bouncing to MENU
        # on a single noisy frame.
        if self._last_state == STATE_IN_MATCH:
            return STATE_IN_MATCH, PATH_HYSTERESIS, banner
        return STATE_MENU, PATH_DEFAULT_MENU, banner

    @staticmethod
    def _elixir_bar_visible(frame: np.ndarray) -> tuple[bool, float]:
        """``(bar visible, magenta fraction)``. The fraction is the number
        the in-match gate turns on, so it is returned rather than
        discarded — a below-threshold frame is meaningless to diagnose
        without knowing how far below it fell."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = ELIXIR_BAR_PATCH
        patch = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if patch.size == 0:
            return False, 0.0
        b = patch[..., 0].astype(int)
        g = patch[..., 1].astype(int)
        r = patch[..., 2].astype(int)
        magenta = (b > 180) & (r > 180) & (g < 140)
        frac = float(magenta.mean())
        return frac >= ELIXIR_MAGENTA_MIN_FRAC, frac

    # ----- side effects -----

    def _click_frac(self, monitor: dict, frac_xy: tuple[float, float]) -> None:
        x = monitor["left"] + int(frac_xy[0] * monitor["width"])
        y = monitor["top"] + int(frac_xy[1] * monitor["height"])
        pyautogui.moveTo(x, y)
        pyautogui.click()

    def _click_template(
        self,
        monitor: dict,
        frame: np.ndarray | None,
        key: str,
        fallback_frac: tuple[float, float],
    ) -> None:
        """Click the template's matched centre, or ``fallback_frac``."""
        frac = fallback_frac
        if frame is not None:
            score, loc = self._locate_template(frame, key)
            if loc is not None and score >= TEMPLATE_MATCH_THRESHOLD:
                frac = loc
        self._click_frac(monitor, frac)

    def click_ok(self, monitor: dict, frame: np.ndarray | None = None) -> None:
        """Dismiss the postmatch screen."""
        self._click_template(monitor, frame, "ok_button", OK_BUTTON_FRAC)

    def click_battle(self, monitor: dict, frame: np.ndarray | None = None) -> None:
        """Start a match from the main menu."""
        self._click_template(monitor, frame, "battle_button", BATTLE_BUTTON_FRAC)
