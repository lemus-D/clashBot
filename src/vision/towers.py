"""Read each tower's HP as the length of its HP bar.

This replaces reading the six HP *numbers* with EasyOCR. The numbers cost
~84ms/cycle and were legible in only ~36-54% of frames. Measuring the bar
costs microseconds, and it produces the quantity the observation actually
wanted: ``tower_hp`` was always *normalised* HP (current over maximum), and
a bar's fill fraction IS that ratio, read directly off the screen.

That removes the learned-maximum machinery entirely. Max HP used to be
inferred as the largest reading of the match, because tower level - and so
max HP - varies per account and per opponent. One misread of 5147 was
therefore enough to inflate the denominator 3.4x and deflate every later
value for the rest of the match. A bar is full when it is full; there is
nothing to learn and nothing to corrupt.

Validated against known values on a real frame: three full-HP towers read
1.000, 1.000 and 1.000; two destroyed towers read 0.000; and a king at
1000/2532 HP read 0.340 against an expected 0.395, the residual being an
uncalibrated bar width. Resolution is ~2% per pixel, or ~30-35 distinct
levels per tower - far finer than the reward needs.

Fill is measured COLUMN-wise: a column counts as filled if any pixel in it
is bar colour, and the fraction is the rightmost filled column over the bar
width. Two reasons rather than taking the area of bar-coloured pixels.
Bars deplete right-to-left, so the position of the boundary is the whole
signal; and an area fraction is diluted by any vertical calibration slop,
where a column measure simply ignores rows that miss the bar.

A destroyed tower reads 0.000, because the game draws no bar at all over
rubble - and so does a bar completely hidden behind a fight. The two are
NOT distinguishable from one frame: what separates them is duration, since
occlusion lasts seconds and destruction lasts the match. This module
therefore reports "no bar visible" as ``None`` and leaves the inference to
``GameState``, which debounces a run of them and retracts if the tower
reads again.

Two discriminators were measured and REJECTED, recorded here so they are
not retried:

- The magenta predicate from ``lifecycle._elixir_bar_visible`` scores 0.000
  on a tower bar. It is tuned for the elixir bar's particular pink.
- Glyph ink fraction from ``hud.glyph_bool`` is not an absence signal. A
  DESTROYED ``friendly_left`` scored 0.376 against 0.152 and 0.189 for two
  LIVE towers, because its rubble sits on a pale wooden bridge that is
  exactly the "light and near-neutral" the glyph gate selects for.

Detecting the bar's empty TRACK - which would separate a live low-HP tower
from a destroyed one in a single frame - was also tried and did not
survive: a king at 60% depletion showed no track under a dark-and-desaturated
predicate (its empty portion is dark at V=71 but too saturated), while a
destroyed tower showed false track hits off a wooden bridge. Worth
revisiting only with frames of a genuinely low-HP princess tower, which the
evidence so far lacks.

``TOWER_BAR_REGIONS`` boxes the bars, NOT the HP numbers. Those are
different places: the princess number is drawn above the bar over open
grass, and the offset between them differs per tower. Do not reuse the old
``TOWER_HP_REGIONS`` values here.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..game.state import TOWER_KEYS
from .hud import crop_region

# (x_frac, y_frac, w_frac, h_frac) of each tower's HP BAR within the
# captured viewport. CALIBRATE FOR YOUR RESOLUTION with
# 'python -m src.main --calibrate towers'.
#
# Width matters as much as position: the fraction is relative to the BOX, so
# a box wider than the bar reads a full tower as damaged. Bound the bar
# tightly left-to-right.
#
# These values are calibrated for this machine and verified against a frame
# with known HP: the three full towers read 1.000 and the two destroyed ones
# 0.000, while a king at 1000/2532 read 0.374 against an expected 0.395 - a
# residual inside the ~2%-per-pixel resolution, not a calibration error.
TOWER_BAR_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "enemy_king":  (0.4483, 0.0277, 0.1494, 0.0194),
    "enemy_left":  (0.2036, 0.1470, 0.1100, 0.0092),
    "enemy_right":  (0.7258, 0.1460, 0.1100, 0.0102),
    "friendly_king":  (0.4483, 0.7505, 0.1494, 0.0176),
    "friendly_left":  (0.2036, 0.6174, 0.1117, 0.0129),
    "friendly_right":  (0.7258, 0.6174, 0.1100, 0.0129),
}

# Hue ranges (OpenCV 0-179) of the two sides' bar fill, measured on real
# pixels: enemy bars are pink at BGR ~(88,36,212), friendly are blue at BGR
# ~(209,176,110). Nothing else near these boxes occupies either range - the
# arena floor is green at hue ~36 and the wooden bridges are ~21-30.
ENEMY_BAR_HUE = (160, 179)
FRIENDLY_BAR_HUE = (95, 115)

# A pixel counts as bar fill only if it is also saturated and bright.
# Rubble, grass and shadow all fail one of these even when their hue
# happens to land in range.
BAR_MIN_SATURATION = 140
BAR_MIN_VALUE = 120


def _hue_range(key: str) -> tuple[int, int]:
    return ENEMY_BAR_HUE if key.startswith("enemy") else FRIENDLY_BAR_HUE


def bar_fill_fraction(frame: np.ndarray, key: str) -> float:
    """How far ``key``'s HP bar is filled, 0.0-1.0.

    ``0.0`` means no bar colour anywhere in the box - a destroyed tower, or
    one whose bar is completely hidden. Returned as a plain float rather
    than thresholded so the debug overlay and the calibration phase can
    show how far a frame sits from a decision; a bare boolean is not
    diagnosable.
    """
    strip = crop_region(frame, TOWER_BAR_REGIONS[key],
                        f"TOWER_BAR_REGIONS[{key!r}]")
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(int)
    sat = hsv[..., 1].astype(int)
    val = hsv[..., 2].astype(int)
    lo, hi = _hue_range(key)
    fill = (
        (hue >= lo) & (hue <= hi)
        & (sat >= BAR_MIN_SATURATION)
        & (val >= BAR_MIN_VALUE)
    )
    filled_columns = np.nonzero(fill.any(axis=0))[0]
    if filled_columns.size == 0:
        return 0.0
    return float((int(filled_columns.max()) + 1) / fill.shape[1])


class TowerBarReader:
    """Reads normalised HP for the six towers from their HP bars.

    :meth:`read` returns ``{tower_key: fill}`` where ``fill`` is 0.0-1.0,
    or ``None`` for a tower whose bar is not visible at all. ``None`` is
    the same "could not read this frame" signal the OCR reader used to
    produce, so ``GameState`` handles it unchanged: keep the last known
    value, and infer destruction only from a sustained run.

    Stateless and cheap - six small crops and an HSV threshold, no model,
    no GPU, nothing to load or release. :meth:`close` exists only so this
    drops in where the OCR readers were.
    """

    def read(self, frame: np.ndarray) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {}
        for key in TOWER_KEYS:
            fill = bar_fill_fraction(frame, key)
            out[key] = fill if fill > 0.0 else None
        return out

    def fractions(self, frame: np.ndarray) -> dict[str, float]:
        """Raw fill fractions for all six towers, un-thresholded."""
        return {key: bar_fill_fraction(frame, key) for key in TOWER_KEYS}

    def close(self) -> None:
        """No-op; kept so this drops in where the OCR readers were."""

    def __enter__(self) -> TowerBarReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
