"""Shared primitives for reading numbers off the game HUD.

Two things every HUD reader in this package needs, factored out of
``ocr.py`` when the elixir reader started needing the same treatment:

- :func:`crop_region` — cut a ``(x_frac, y_frac, w_frac, h_frac)`` box out
  of a captured frame. Regions are stored as fractions so one set of
  CALIBRATE constants survives a window resize.
- :func:`glyph_bool` / :func:`glyph_mask` — isolate the near-white glyphs
  the game draws over its saturated HUD elements.

The glyph isolation is the load-bearing part and it is not a grayscale
threshold, for a measured reason: the elements these digits sit on are
glossy and saturated - magenta for enemy tower bars, blue for friendly,
purple for the elixir counter - and at the highlight they are both bright
AND pale, so brightness alone floods the mask. Working in LAB and keeping
pixels that are light AND near-neutral picks out the glyphs and nothing
else, because chroma is distance from the neutral axis: the bar fill, the
gold level badge and the arena floor are all strongly chromatic and the
glyphs are not. Lightness is taken as a percentile so it adapts per crop
instead of needing a per-element constant.

Polarity is fixed by construction rather than voted on from pixel counts,
so it cannot flip between neighbouring frames - which is what ruled out
the Otsu-majority approach still used for the match timer, whose crops sit
at 39-48% ink, close enough to the flip point that it inverted
inconsistently.
"""

from __future__ import annotations

import cv2
import numpy as np

# Upscale applied before the LAB conversion. The HUD digits are ~14 px
# tall; recognisers and template matching both do measurably better with
# more pixels to work with.
GLYPH_UPSCALE = 4.0

# A pixel is glyph if its lightness is at or above this percentile of the
# crop AND its chroma is at or below GLYPH_MAX_CHROMA.
GLYPH_LIGHTNESS_PERCENTILE = 55.0
GLYPH_MAX_CHROMA = 18.0

# Background value in the returned uint8 mask. Glyphs are 0 (black on
# white) because that is the polarity Tesseract is trained on.
GLYPH_BACKGROUND = 255


def crop_region(
    frame: np.ndarray,
    region: tuple[float, float, float, float],
    what: str,
    margin: int = 0,
) -> np.ndarray:
    """Crop ``(x_frac, y_frac, w_frac, h_frac)`` out of ``frame``.

    ``what`` names the calibration constant the region came from, so an
    empty crop says which value to fix. ``margin`` grows the box by that
    many source pixels on every side.
    """
    h, w = frame.shape[:2]
    xf, yf, wf, hf = region
    x0 = max(0, int(xf * w) - margin)
    y0 = max(0, int(yf * h) - margin)
    x1 = min(w, int(xf * w) + max(1, int(wf * w)) + margin)
    y1 = min(h, int(yf * h) + max(1, int(hf * h)) + margin)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(
            f"{what} produced an empty crop ({x0},{y0})-({x1},{y1}) from a "
            f"{w}x{h} frame; check its calibration"
        )
    return crop


def glyph_bool(crop: np.ndarray) -> np.ndarray:
    """Boolean mask, ``True`` where ``crop`` holds a near-white glyph.

    Nothing here special-cases an absent number. A crop with no digits in
    it - a destroyed tower, an undamaged king, a menu screen - simply
    yields few or no True pixels, and the caller decides what that means.
    """
    big = cv2.resize(crop, None, fx=GLYPH_UPSCALE, fy=GLYPH_UPSCALE,
                     interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    return (
        (lightness >= np.percentile(lightness, GLYPH_LIGHTNESS_PERCENTILE))
        & (chroma <= GLYPH_MAX_CHROMA)
    )


def glyph_mask(crop: np.ndarray) -> np.ndarray:
    """:func:`glyph_bool` as a uint8 image: black glyphs on white.

    This is the form the OCR recognisers want. Template matching wants the
    boolean instead - see :func:`glyph_bool`.
    """
    return np.where(glyph_bool(crop), 0, GLYPH_BACKGROUND).astype(np.uint8)
