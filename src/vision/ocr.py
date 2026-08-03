"""OCR for the match timer - the one HUD element still read by a recogniser.

``MatchTimerReader`` uses Tesseract, which reads the ``m:ss`` countdown
fine. This is the only ground truth for match time, everything else about
the clock being simulated. It keeps its own persistent ``PyTessBaseAPI``
so tessdata loads once rather than per cycle.

This module used to also read the six tower HP numbers, with EasyOCR after
Tesseract proved unable to (it plateaued at 9 of 12 known readings while
emitting confident wrong values like 112 or 1812 for 1512, and one misread
0 for ``enemy_king`` ended a match at 2:28 with a false "win"). EasyOCR
did better but still only read ~36-54% of frames and cost ~84 ms/cycle.
Both are gone: tower HP is now the fill fraction of the on-screen HP bar,
measured geometrically in ``towers.py`` for microseconds and with no
recogniser to be wrong. The elixir count moved to template matching in
``elixir.py`` for the same reason.

The region is a fraction of the captured frame (0-1); CALIBRATE
``MATCH_TIMER_REGION`` for your BlueStacks crop.

Tesseract is a hard requirement: a missing install or a malformed crop
raises rather than degrading silently. A single frame whose digits cannot
be read is not a failure of that kind - it is reported as ``None`` and the
caller decides.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tesserocr import PSM, RIL, PyTessBaseAPI

from .hud import crop_region


# (x_frac, y_frac, w_frac, h_frac) of the "m:ss" match countdown, drawn in
# the upper right of the arena view, roughly level with the enemy king
# tower's HP bar. CALIBRATE FOR YOUR RESOLUTION - this is a starting guess
# only, and a wrong region makes the match clock unanchorable (which
# ``ClashEnv`` raises about rather than running on a simulated clock).
MATCH_TIMER_REGION: tuple[float, float, float, float] = (0.8555, 0.0240, 0.1248, 0.0370)


# The timer crop holds exactly one line, so psm 7 skips layout analysis
# entirely. The colon is whitelisted because it is part of the value;
# without a whitelist Tesseract readily returns O for 0 and l/I for 1.
MATCH_TIMER_PSM = PSM.SINGLE_LINE
MATCH_TIMER_WHITELIST = "0123456789:"

# Longest countdown either the regulation or the overtime timer can show
# (regulation starts at 3:00, overtime restarts at 2:00). Anything larger
# is a misread, not a clock.
MAX_TIMER_SECONDS = 180

# The timer crop is binarized to black digits on a white background.
# Dark-on-light rather than the inverse because Tesseract is trained that
# way: measured over 2304 synthetic region reads the two conventions were
# identical on accuracy (2226 correct either way) but dark-on-light read in
# 40.2 ms against 58.5 ms.
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


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Binarize the timer crop to black digits on a white background.

    Orientation comes from the pixel counts: the digits occupy a small
    minority of this tight box, so the majority class after Otsu is
    background and is forced to white. Should the crop ever break that
    assumption its digits invert and the frame reads as None.

    A majority-vote polarity is only safe while the digits are a clear
    minority of the crop, which is true of this tight one-line box. It is
    not universally true of HUD crops - the old tower-HP boxes sat at
    39-48% ink, close enough to the flip point that neighbouring frames
    inverted inconsistently, which is why ``hud.glyph_mask`` fixes polarity
    by construction instead.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Upscale + threshold makes Tesseract substantially more reliable on
    # the small UI digits.
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    if gray.min() == gray.max():
        # A flat crop - the timer is absent in the menu and behind the
        # pre-match countdown - has no foreground to find, and Otsu on a
        # constant image is degenerate. Answer with pure background so the
        # frame reads as None.
        return np.full_like(gray, _BACKGROUND)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if 2 * int(np.count_nonzero(binarized)) < binarized.size:
        binarized = cv2.bitwise_not(binarized)
    return binarized


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

    Owns one persistent Tesseract API handle and is NOT thread-safe: read
    from one thread only, or give each thread its own reader.
    """

    def __init__(self) -> None:
        self._api: PyTessBaseAPI | None = None

    def read(self, frame: np.ndarray) -> float | None:
        crop = _preprocess_for_ocr(
            crop_region(frame, MATCH_TIMER_REGION, "MATCH_TIMER_REGION")
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
