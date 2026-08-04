"""Count the crowns each side won, from the postmatch screen.

``MatchLifecycle`` reads the postmatch banner for win/loss/draw, which says
who won but not by how much. This reads the crown row for each side, so a
3-0 can be told from a 1-0.

It is also the only TRUSTWORTHY crown count in the program. During a match
``GameState`` infers crowns from tower-destruction debouncing, which is
absence-based and drifts: a fight covering a tower's HP bar registers a
destruction that is later retracted, and the running tally moves with it (a
real run reached 2-1 before handing both back). This screen is what a human
reads, so ``GameState.set_final_crowns`` lets it overwrite that guess.

Read once per match, on the postmatch transition - not per cycle.

**Crowns are counted as connected gold blobs, not by testing fixed slot
positions.** The winner's row is drawn in a taller band than the loser's, so
slot geometry is not shared between the two sides and would have to be
calibrated per outcome. Blob counting does not care: measured on a real 2-1
victory screen, crown blobs were 5884-6343 px against a largest non-crown
blob of 754 px, so the two populations are an order of magnitude apart.

The rows do not move: the friendly (blue) side is always the bottom row and
the enemy the top, so each region simply *is* one side's row.

Cushion colour is still checked, but as a guard rather than to work out
which row is which. An empty crown slot shows a cushion in its side's own
colour - magenta for the enemy, blue for the friendly side, the same
convention as the in-match HP bars and the detector's blue/red class naming
- and a row that does not show its expected colour means this frame is not
the postmatch screen. That is what stops a crown score being read off, say,
an in-match frame if the lifecycle state is ever wrong: measured, the enemy
row is 25.0% magenta against 0.3% blue and the friendly row 31.4% blue
against 0.0% magenta, while an in-match frame matches neither.

Both regions are CALIBRATE. Box each crown row snugly: a crown blob has to
clear ``MIN_BLOB_AREA_FRAC`` of the region's area, so a needlessly tall box
shrinks every blob's share and can push real crowns under the threshold.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .hud import crop_region

logger = logging.getLogger(__name__)

# Each side's crown row on the postmatch screen, as
# (x_frac, y_frac, w_frac, h_frac) of the captured viewport. The enemy row is
# the top one and the friendly row the bottom one - the layout does not
# change, so the side is a property of the region rather than something to
# infer per frame.
# CALIBRATE FOR YOUR RESOLUTION with 'python -m src.main --calibrate crowns'.
CROWN_ROW_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "enemy": (0.0854, 0.1479, 0.8013, 0.1959),
    "friendly": (0.1002, 0.4427, 0.7800, 0.1155),
}

# Gold crown fill. Measured on real pixels at hue 14-20 - noticeably more
# orange than a "yellow" guess, which is why a 18-32 range missed most of
# each crown.
GOLD_HUE = (8, 26)
GOLD_MIN_SATURATION = 140
GOLD_MIN_VALUE = 110

# Empty-slot cushion colours, used to confirm a row really is that side's
# crown row on a postmatch screen (see MIN_CUSHION_FRAC).
ENEMY_CUSHION_HUE = (160, 179)
FRIENDLY_CUSHION_HUE = (95, 118)
CUSHION_MIN_SATURATION = 120
CUSHION_MIN_VALUE = 80

# Minimum share of a row that must show its own side's cushion colour for the
# frame to be accepted as the postmatch screen. Measured at 25-31% on a real
# victory screen against 0% for the wrong side's colour and for an in-match
# frame, so this sits far below the real signal and far above the noise. It
# only has to reject "this is not the postmatch screen at all".
MIN_CUSHION_FRAC = 0.05

# A gold blob counts as a crown at or above this share of its row region's
# area. Deliberately well below the smallest real crown rather than midway
# between crown and noise, because the share depends on how tightly the row
# was boxed and the two rows are not boxed alike: on the calibrated regions
# the friendly crowns are 8.9-10.0% of their box while the enemy crown is
# 5.85% of its noticeably taller one. Non-crown blobs measured 0.37-0.66%,
# so this leaves real crowns 3-5x clear of the threshold and noise 3x below
# it, instead of a 1.5x margin on whichever row happens to be boxed loosest.
MIN_BLOB_AREA_FRAC = 0.02

# Closing kernel, to join a crown split by its own dark outline or by a
# highlight before blobs are counted.
_CLOSE_KERNEL = np.ones((5, 5), np.uint8)

MAX_CROWNS_PER_SIDE = 3


def _mask(hsv: np.ndarray, hue: tuple[int, int], min_sat: int, min_val: int) -> np.ndarray:
    h = hsv[..., 0].astype(int)
    s = hsv[..., 1].astype(int)
    v = hsv[..., 2].astype(int)
    return (h >= hue[0]) & (h <= hue[1]) & (s >= min_sat) & (v >= min_val)


def count_crowns(region_bgr: np.ndarray) -> int:
    """Number of gold crowns in one crown-row crop."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    gold = _mask(hsv, GOLD_HUE, GOLD_MIN_SATURATION, GOLD_MIN_VALUE)
    closed = cv2.morphologyEx(gold.astype(np.uint8), cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    min_area = MIN_BLOB_AREA_FRAC * region_bgr.shape[0] * region_bgr.shape[1]
    crowns = sum(
        1 for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    )
    return min(crowns, MAX_CROWNS_PER_SIDE)


def cushion_fraction(region_bgr: np.ndarray, side: str) -> float:
    """Share of ``region_bgr`` showing ``side``'s cushion colour.

    Used to confirm a region really is that side's crown row on a postmatch
    screen. Returned as the raw fraction so calibration can show how far a
    frame sits from the threshold.
    """
    hue = ENEMY_CUSHION_HUE if side == "enemy" else FRIENDLY_CUSHION_HUE
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    return float(_mask(hsv, hue, CUSHION_MIN_SATURATION,
                       CUSHION_MIN_VALUE).mean())


class PostMatchCrownReader:
    """Reads ``{"friendly": n, "enemy": m}`` off the postmatch screen.

    :meth:`read` returns ``None`` when a row does not show its side's
    cushion colour, which means the frame is not the postmatch screen (or
    the regions are miscalibrated). Returning ``None`` rather than a guess
    matters here because the caller uses this to OVERWRITE its own crown
    tally and to scale the terminal reward; a wrong count read off the
    wrong screen would be worse than keeping the inferred one.

    Stateless and cheap: two crops and a couple of HSV thresholds.
    """

    def read(self, frame: np.ndarray) -> Optional[dict[str, int]]:
        counts: dict[str, int] = {}
        for side, region in CROWN_ROW_REGIONS.items():
            crop = crop_region(frame, region, f"CROWN_ROW_REGIONS[{side!r}]")
            cushion = cushion_fraction(crop, side)
            if cushion < MIN_CUSHION_FRAC:
                logger.warning(
                    "Postmatch crown row %r shows only %.1f%% of its expected "
                    "cushion colour (needs %.0f%%), so this frame is not the "
                    "postmatch screen or CROWN_ROW_REGIONS is miscalibrated. "
                    "Not reading crowns.",
                    side, cushion * 100, MIN_CUSHION_FRAC * 100,
                )
                return None
            counts[side] = count_crowns(crop)
        return counts

    def fractions(self, frame: np.ndarray) -> dict[str, dict]:
        """Per-row diagnostics: cushion fraction and crown count. For the
        calibration phase and debugging."""
        out: dict[str, dict] = {}
        for side, region in CROWN_ROW_REGIONS.items():
            crop = crop_region(frame, region, f"CROWN_ROW_REGIONS[{side!r}]")
            out[side] = {
                "cushion_frac": round(cushion_fraction(crop, side), 4),
                "crowns": count_crowns(crop),
            }
        return out

    def close(self) -> None:
        """No-op; kept for symmetry with the other readers."""
