"""OCR readers for the two numeric HUD elements: tower HP and the match timer.

``TowerHealthReader`` crops a small region around each of the six towers
and runs Tesseract on the digits. ``MatchTimerReader`` does the same for
the single ``m:ss`` countdown in the arena's upper right, which is the
only ground truth for match time (everything else about the clock is
simulated). Regions are fractions of the captured frame (0-1); CALIBRATE
``TOWER_HP_REGIONS`` and ``MATCH_TIMER_REGION`` for your BlueStacks crop.

Tesseract runs in-process via tesserocr, which binds libtesseract
directly. One ``PyTessBaseAPI`` is built lazily and reused for the life
of the reader, so tessdata is loaded once instead of once per cycle.
(The pytesseract path this replaced spawned a fresh ``tesseract.exe``
per call: ~200 ms of process overhead against ~35 ms of recognition.)

The six crops are still stacked into one image and read in a single
recognition pass. Measured with a persistent handle it is the faster of
the two: 51 ms/cycle against 61 ms for six separate recognitions, because
Tesseract's fixed per-image layout analysis costs more than the blank
padding between rows. Each recognised word is mapped back to its source
region by its vertical position in the stack.

Because the crops share one canvas they must share one polarity, so
``_preprocess_for_ocr`` normalises every crop to black digits on white
independently of the others. That invariant is what makes compositing
safe: deriving the canvas polarity from the crops collectively (a
majority vote over their pixels) let a mostly-destroyed board - three
blank crops outvoting three live ones, an ordinary late-game state -
invert the canvas and silently erase the survivors' digits.

Tesseract is a hard requirement: a missing install, missing tessdata, or
a malformed crop all raise rather than degrading silently. A single frame
whose digits cannot be recognised is not a failure of that kind - it is
reported as ``None`` for that field and the caller decides.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tesserocr import PSM, RIL, PyTessBaseAPI

from ..game.state import TOWER_KEYS


# Each entry is (x_frac, y_frac, w_frac, h_frac) within the captured frame.
# CALIBRATE FOR YOUR RESOLUTION.
TOWER_HP_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "enemy_king":  (0.3974, 0.0120, 0.2102, 0.0444),
    "enemy_left":  (0.1658, 0.1322, 0.1527, 0.0351),
    "enemy_right":  (0.6864, 0.1331, 0.1527, 0.0333),
    "friendly_king":  (0.3924, 0.7403, 0.2200, 0.0471),
    "friendly_left":  (0.1626, 0.6118, 0.1576, 0.0582),
    "friendly_right":  (0.6880, 0.6091, 0.1494, 0.0370),
}

# (x_frac, y_frac, w_frac, h_frac) of the "m:ss" match countdown, drawn in
# the upper right of the arena view, roughly level with the enemy king
# tower's HP bar. CALIBRATE FOR YOUR RESOLUTION - this is a starting guess
# only, and a wrong region makes the match clock unanchorable (which
# ``ClashEnv`` raises about rather than running on a simulated clock).
MATCH_TIMER_REGION: tuple[float, float, float, float] = (0.8555, 0.0240, 0.1248, 0.0370)


# PSM.SINGLE_BLOCK (psm 6) = "a uniform block of text": Tesseract segments
# the stack into one line per crop. Tower HP is always digits.
TOWER_HP_PSM = PSM.SINGLE_BLOCK
TOWER_HP_WHITELIST = "0123456789"

# The timer crop holds exactly one line, so psm 7 skips layout analysis
# entirely. The colon is whitelisted because it is part of the value;
# without a whitelist Tesseract readily returns O for 0 and l/I for 1.
MATCH_TIMER_PSM = PSM.SINGLE_LINE
MATCH_TIMER_WHITELIST = "0123456789:"

# Longest countdown either the regulation or the overtime timer can show
# (regulation starts at 3:00, overtime restarts at 2:00). Anything larger
# is a misread, not a clock.
MAX_TIMER_SECONDS = 180

# Blank rows between stacked crops, and a border around the whole stack,
# so Tesseract splits the crops into separate lines. Not per-machine:
# the crops are already 3x upscaled, so these are generic layout padding.
_ROW_GAP_PX = 24
_MARGIN_PX = 12

# Every preprocessed crop is black digits on a white background, so this is
# also what the composite is padded with. Dark-on-light rather than the
# inverse because Tesseract is trained that way: measured over 2304 synthetic
# region reads the two conventions were identical on accuracy (2226 correct
# either way) but dark-on-light read in 40.2 ms against 58.5 ms.
_BACKGROUND = 255


def _resolve_tessdata_dir() -> str:
    """Locate the tessdata directory holding ``eng.traineddata``.

    tesserocr bundles libtesseract but not the language data, so the
    directory has to be found explicitly. Honours ``TESSDATA_PREFIX``
    (Tesseract 5 expects it to *be* the tessdata dir, Tesseract 4 expected
    its parent - both are accepted), else falls back to the ``tessdata``
    beside the ``tesseract`` executable on PATH.
    """
    candidates: list[Path] = []
    prefix = os.environ.get("TESSDATA_PREFIX")
    if prefix:
        candidates += [Path(prefix), Path(prefix) / "tessdata"]
    executable = shutil.which("tesseract")
    if executable:
        candidates.append(Path(executable).resolve().parent / "tessdata")

    for candidate in candidates:
        if (candidate / "eng.traineddata").is_file():
            return str(candidate)

    raise RuntimeError(
        "Cannot locate Tesseract language data: no eng.traineddata in "
        f"{[str(c) for c in candidates] or 'any candidate directory'}. "
        "Install Tesseract-OCR (so 'tesseract' is on PATH) or point "
        "TESSDATA_PREFIX at a directory containing eng.traineddata."
    )


def _make_api(psm: PSM, whitelist: str) -> PyTessBaseAPI:
    """Build a Tesseract handle restricted to ``whitelist`` at ``psm``."""
    tessdata = _resolve_tessdata_dir()
    try:
        api = PyTessBaseAPI(path=tessdata, lang="eng", psm=psm)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Tesseract failed to initialise from tessdata {tessdata!r}: "
            f"{exc}. OCR is required; check the Tesseract install and that "
            "eng.traineddata matches its version."
        ) from exc
    api.SetVariable("tessedit_char_whitelist", whitelist)
    return api


def _recognized_words(api: PyTessBaseAPI) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Recognised words as (text, bounding box) pairs.

    An all-blank image - every tower destroyed or obscured, or a timer
    hidden behind the pre-match countdown, both legitimate - leaves the
    iterator empty, and tesserocr raises "No text returned" if that is read
    anyway. Skip empty elements so the caller gets Nones instead of an
    exception.
    """
    iterator = api.GetIterator()
    if iterator is None:
        return []
    words: list[tuple[str, tuple[int, int, int, int]]] = []
    iterator.Begin()
    while True:
        if not iterator.Empty(RIL.WORD):
            box = iterator.BoundingBox(RIL.WORD)
            if box is not None:
                words.append((iterator.GetUTF8Text(RIL.WORD), box))
        if not iterator.Next(RIL.WORD):
            return words


def _crop_region(
    frame: np.ndarray, region: tuple[float, float, float, float], what: str
) -> np.ndarray:
    """Crop ``(x_frac, y_frac, w_frac, h_frac)`` out of ``frame``.

    ``what`` names the calibration constant the region came from, so an
    empty crop says which value to fix.
    """
    h, w = frame.shape[:2]
    xf, yf, wf, hf = region
    x0 = max(0, int(xf * w))
    y0 = max(0, int(yf * h))
    x1 = min(w, x0 + max(1, int(wf * w)))
    y1 = min(h, y0 + max(1, int(hf * h)))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(
            f"{what} produced an empty crop ({x0},{y0})-({x1},{y1}) from a "
            f"{w}x{h} frame; check its calibration"
        )
    return crop


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Binarize one small HUD crop to black digits on a white background.

    Polarity is decided from this crop alone, never from its neighbours, so
    that stacking crops cannot make one region's contents change another's
    rendering. Orientation comes from the pixel counts: digits occupy a
    small minority of one of these tight boxes - 12-22% of pixels across
    the six TOWER_HP_REGIONS on synthetic renders, and MATCH_TIMER_REGION is
    a box of the same kind around four glyphs - so the majority class after
    Otsu is background and is forced to white. Should some region ever
    break that assumption its own digits invert and it reads as None; it
    cannot drag the other five with it, which is the point of deciding per
    crop.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Upscale + threshold makes Tesseract substantially more reliable on
    # the small UI digits.
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    if gray.min() == gray.max():
        # A flat crop - a destroyed tower leaves no HP label - has no
        # foreground to find, and Otsu on a constant image is degenerate.
        # Answer directly with pure background: this region reads as None
        # and contributes nothing to the composite.
        return np.full_like(gray, _BACKGROUND)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if 2 * int(np.count_nonzero(binarized)) < binarized.size:
        binarized = cv2.bitwise_not(binarized)
    return binarized


def _stack_crops(crops: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    """Stack preprocessed crops vertically into one OCR image.

    Every crop arrives from :func:`_preprocess_for_ocr` in the same
    polarity, so the padding is just ``_BACKGROUND`` and no crop's contents
    can affect how another is rendered.

    Returns the composite and the y coordinates separating consecutive
    crops, used to attribute recognised words back to their region.
    """
    width = max(c.shape[1] for c in crops) + 2 * _MARGIN_PX
    height = sum(c.shape[0] for c in crops) + _ROW_GAP_PX * (len(crops) + 1)
    canvas = np.full((height, width), _BACKGROUND, dtype=np.uint8)
    boundaries: list[float] = []
    y = _ROW_GAP_PX
    for i, crop in enumerate(crops):
        canvas[y:y + crop.shape[0], _MARGIN_PX:_MARGIN_PX + crop.shape[1]] = crop
        y += crop.shape[0]
        if i < len(crops) - 1:
            boundaries.append(y + _ROW_GAP_PX / 2.0)
        y += _ROW_GAP_PX
    return canvas, boundaries


class TowerHealthReader:
    """Reads tower HP for the six towers via Tesseract OCR.

    Owns one persistent Tesseract API handle, created on the first
    :meth:`read` and released by :meth:`close`. NOT thread-safe: a
    ``PyTessBaseAPI`` wraps a single stateful ``TessBaseAPI``, so
    concurrent reads would interleave ``SetImageBytes`` / ``Recognize``
    on the same instance. Call :meth:`read` from one thread only, or give
    each thread its own reader.
    """

    def __init__(self) -> None:
        self._api: PyTessBaseAPI | None = None

    def read(self, frame: np.ndarray) -> dict[str, int | None]:
        keys = list(TOWER_HP_REGIONS)
        crops = [
            _preprocess_for_ocr(
                _crop_region(frame, TOWER_HP_REGIONS[k], f"TOWER_HP_REGIONS[{k!r}]")
            )
            for k in keys
        ]
        composite, boundaries = _stack_crops(crops)

        api = self._get_api()
        api.SetImageBytes(
            composite.tobytes(),
            composite.shape[1],
            composite.shape[0],
            1,                    # bytes per pixel: 8-bit grayscale
            composite.shape[1],   # bytes per line
        )
        api.Recognize()

        # Attribute each word to a crop by where it sits vertically. Position
        # rather than line order: a tower whose HP is unreadable (destroyed,
        # obscured) must yield None for *that* tower, never shift the
        # remaining readings onto the wrong towers.
        found: dict[int, list[tuple[int, str]]] = {}
        for text, (left, top, _right, bottom) in _recognized_words(api):
            text = text.strip()
            if not text:
                continue
            idx = int(np.searchsorted(boundaries, (top + bottom) / 2.0))
            found.setdefault(idx, []).append((left, text))

        out: dict[str, int | None] = {k: None for k in TOWER_KEYS}
        for i, key in enumerate(keys):
            words = found.get(i)
            if not words:
                continue
            text = " ".join(t for _, t in sorted(words))
            if text.isdigit():
                out[key] = int(text)
        return out

    def close(self) -> None:
        """Release the Tesseract handle. Idempotent; a later :meth:`read`
        transparently builds a fresh one."""
        if self._api is not None:
            self._api.End()
            self._api = None

    def __enter__(self) -> TowerHealthReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----- internals -----

    def _get_api(self) -> PyTessBaseAPI:
        if self._api is None:
            self._api = _make_api(TOWER_HP_PSM, TOWER_HP_WHITELIST)
        return self._api


def parse_match_timer(text: str) -> float | None:
    """Parse an on-screen countdown into seconds *remaining*, or None.

    The game draws ``m:ss`` and counts DOWN - 3:00 at the start of
    regulation, and a fresh 2:00 when overtime begins - so the returned
    value is time left on whichever countdown is on screen, never elapsed
    time. Disambiguating the two countdowns is the caller's problem; see
    :class:`MatchTimerReader`.

    Tesseract's failure modes on this glyph run are dropping the thin
    colon and substituting look-alikes, so the accepted forms are narrow
    and everything else is a rejected read (None) rather than a guess:

    - ``"2:47"`` - the normal case.
    - ``"247"`` - colon dropped. Unambiguous because the seconds field is
      always two digits, so a 3-digit run is m + ss.
    - anything else, including 4-digit runs (a colon misread as a digit),
      seconds >= 60, and totals above ``MAX_TIMER_SECONDS``, is None.

    Character substitutions are handled upstream by
    ``MATCH_TIMER_WHITELIST``; whitespace and stray whitelist characters
    are stripped here.
    """
    cleaned = re.sub(r"[^0-9:]", "", text)
    match = re.fullmatch(r"(\d):([0-5]\d)", cleaned) or re.fullmatch(
        r"(\d)([0-5]\d)", cleaned
    )
    if match is None:
        return None
    remaining = float(int(match.group(1)) * 60 + int(match.group(2)))
    if remaining > MAX_TIMER_SECONDS:
        return None
    return remaining


class MatchTimerReader:
    """Reads the on-screen ``m:ss`` match countdown via Tesseract OCR.

    :meth:`read` returns seconds remaining on the displayed countdown, or
    ``None`` when this frame could not be read - timer absent (menu,
    postmatch), occluded by the pre-match countdown overlay, mid-animation,
    or a parse that failed validation. A ``None`` is an ordinary outcome
    for one frame, exactly as an unreadable tower HP is; what an unreadable
    *run* of frames means is the caller's decision.

    Overtime is deliberately NOT disambiguated here. The overtime clock
    restarts from 2:00, so "1:30" alone cannot say whether 90s of
    regulation or 90s of overtime remain, and a reader that guessed would
    hand out an elapsed time two minutes wrong. This class reports the
    displayed countdown as-is; ``GameState.anchor_match_clock`` is what
    rejects a reading that disagrees with its own clock, which is what
    keeps an overtime display from ever anchoring the match clock.

    Owns one persistent Tesseract API handle with the same threading
    caveat as :class:`TowerHealthReader`: one thread per reader.
    """

    def __init__(self) -> None:
        self._api: PyTessBaseAPI | None = None

    def read(self, frame: np.ndarray) -> float | None:
        crop = _preprocess_for_ocr(
            _crop_region(frame, MATCH_TIMER_REGION, "MATCH_TIMER_REGION")
        )
        api = self._get_api()
        api.SetImageBytes(
            crop.tobytes(),
            crop.shape[1],
            crop.shape[0],
            1,                # bytes per pixel: 8-bit grayscale
            crop.shape[1],    # bytes per line
        )
        api.Recognize()
        text = "".join(t.strip() for t, _box in _recognized_words(api))
        if not text:
            return None
        return parse_match_timer(text)

    def close(self) -> None:
        """Release the Tesseract handle. Idempotent; a later :meth:`read`
        transparently builds a fresh one."""
        if self._api is not None:
            self._api.End()
            self._api = None

    def __enter__(self) -> MatchTimerReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----- internals -----

    def _get_api(self) -> PyTessBaseAPI:
        if self._api is None:
            self._api = _make_api(MATCH_TIMER_PSM, MATCH_TIMER_WHITELIST)
        return self._api
