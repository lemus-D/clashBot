"""OCR readers for the two numeric HUD elements: tower HP and the match timer.

The two use different recognisers, for measured reasons.

``TowerHealthReader`` reads the six tower HP numbers with EasyOCR on the
GPU. It used Tesseract, which could not do it: the HP digits are a heavy
stylised game font about 14 px tall, and across ~400 combinations of
threshold, mask, upscale (3-8x), interpolation, morphology, PSM and engine
mode it plateaued at 9 of 12 known readings - while emitting confident
wrong values like 112 or 1812 for 1512. Those flow straight into game
state; a single misread 0 for ``enemy_king`` once ended a match at 2:28
with a false "win". The masks were provably clean (the same images are
trivially legible), so the recogniser was the limit, not preprocessing.

EasyOCR reads the same 12 samples 11 exactly right with ZERO wrong
values: the one it is unsure of scores 0.42 against 0.997-1.000 for every
correct reading, so ``TOWER_HP_MIN_CONFIDENCE`` rejects it. Abstaining
costs one frame at a ~0.3 s cycle; a wrong value corrupts the episode.
That confidence gate is also what makes "no readable number" trustworthy
enough to mean "destroyed". Cost is ~84 ms/cycle for all six against
Tesseract's 51 ms, inside the step budget.

``MatchTimerReader`` still uses Tesseract, which reads the ``m:ss``
countdown fine - it is the only ground truth for match time, everything
else about the clock being simulated. It keeps its own persistent
``PyTessBaseAPI`` so tessdata loads once rather than per cycle.

Regions are fractions of the captured frame (0-1); CALIBRATE
``TOWER_HP_REGIONS`` and ``MATCH_TIMER_REGION`` for your BlueStacks crop.
Tower boxes must bound the HP DIGITS only - excluding the gold level
badge to their left, whose small number otherwise reads as HP (that is
where the 1-77 "HP" values in early recordings came from). Before a king
tower takes damage the game draws no bar and no number, only the badge,
centred where the number would be; that crop is expected to yield
``None`` and leave the tower at its default HP.

Both recognisers are hard requirements: a missing install or a malformed
crop raises rather than degrading silently. A single frame whose digits
cannot be read is not a failure of that kind - it is reported as ``None``
for that field and the caller decides.
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
    "enemy_king":  (0.4778, 0.0166, 0.0788, 0.0213),
    "enemy_left":  (0.2085, 0.1322, 0.0706, 0.0203),
    "enemy_right":  (0.7307, 0.1303, 0.0706, 0.0213),
    "friendly_king":  (0.4762, 0.7560, 0.0821, 0.0222),
    "friendly_left":  (0.2085, 0.6201, 0.0706, 0.0203),
    "friendly_right":  (0.7307, 0.6211, 0.0657, 0.0185),
}

# (x_frac, y_frac, w_frac, h_frac) of the "m:ss" match countdown, drawn in
# the upper right of the arena view, roughly level with the enemy king
# tower's HP bar. CALIBRATE FOR YOUR RESOLUTION - this is a starting guess
# only, and a wrong region makes the match clock unanchorable (which
# ``ClashEnv`` raises about rather than running on a simulated clock).
MATCH_TIMER_REGION: tuple[float, float, float, float] = (0.8555, 0.0240, 0.1248, 0.0370)


# Tower HP is always digits, so EasyOCR is restricted to them.
TOWER_HP_ALLOWLIST = "0123456789"

# Minimum EasyOCR confidence for a tower-HP reading to be believed. Over the
# calibration frames every correct reading scored 0.997-1.000 and the single
# misread scored 0.421, so the gap this sits in is wide rather than tuned.
# Below it the frame is reported unreadable (None) instead of guessed at.
TOWER_HP_MIN_CONFIDENCE = 0.90

# The HP digits are near-white glyphs drawn over a saturated bar - magenta
# for the enemy, blue for the friendly side - beside a gold level badge.
# Selecting "light and near-neutral" in LAB isolates them regardless of what
# is behind them, which a grayscale threshold cannot do: the enemy bar is
# glossy magenta, bright AND pale at the highlight, so brightness alone
# floods the mask. Chroma is the distance from the neutral axis, so the bar,
# the badge and the arena floor are all excluded by it while the glyphs are
# not. Lightness is a percentile so it adapts per crop.
_GLYPH_UPSCALE = 4.0
_GLYPH_LIGHTNESS_PERCENTILE = 55.0
_GLYPH_MAX_CHROMA = 18.0

# Calibrated tower boxes bound the digits tightly enough to clip their tops
# and bottoms, and a glyph cut off at the border recognises badly (1512 read
# as 52). A couple of source pixels of slack fixes that. Deliberately small:
# the gold level badge sits ~26 px to the left, so a generous margin trades
# clipped glyphs for a badge digit in the crop.
_TOWER_CROP_MARGIN_PX = 2

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


def _crop_region(
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


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Binarize the timer crop to black digits on a white background.

    Orientation comes from the pixel counts: the digits occupy a small
    minority of this tight box, so the majority class after Otsu is
    background and is forced to white. Should the crop ever break that
    assumption its digits invert and the frame reads as None.

    Tower HP does NOT come through here - see :func:`_glyph_mask`. A
    majority-vote polarity is only safe while the digits are a clear
    minority of the crop, and the tower boxes sit at 39-48% ink, close
    enough to the flip point that neighbouring frames inverted
    inconsistently.
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


def _glyph_mask(crop: np.ndarray) -> np.ndarray:
    """Isolate tower-HP digits as black glyphs on a white background.

    Works in LAB and keeps pixels that are light AND near-neutral, which is
    what the digits are and what nothing else in these crops is: the bar
    fill and the gold level badge are strongly chromatic, and the dark glyph
    outline is not light. Polarity is fixed by construction rather than
    voted on, so it cannot flip between frames.

    Nothing here special-cases a destroyed tower or an undamaged king. Both
    simply contain no digits, the recogniser finds nothing or scores low,
    and the caller gets ``None``.
    """
    big = cv2.resize(crop, None, fx=_GLYPH_UPSCALE, fy=_GLYPH_UPSCALE,
                     interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    lightness = lab[..., 0].astype(np.float32)
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    glyph = (
        (lightness >= np.percentile(lightness, _GLYPH_LIGHTNESS_PERCENTILE))
        & (chroma <= _GLYPH_MAX_CHROMA)
    )
    return np.where(glyph, 0, _BACKGROUND).astype(np.uint8)


def _make_easyocr_reader():
    """Build the EasyOCR reader used for tower HP.

    Imported here rather than at module scope because EasyOCR pulls in
    torch, and ``src.calibrate`` imports this module only for the region
    constants - it should not pay seconds of CUDA init to draw boxes.
    """
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR is required to read tower HP but is not installed: "
            f"{exc}. Install it with 'pip install easyocr' (see "
            "requirements.txt)."
        ) from exc
    try:
        return easyocr.Reader(["en"], gpu=True, verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"EasyOCR failed to initialise: {exc}. Tower HP cannot be read; "
            "check the torch install and that the recognition model "
            "downloaded to ~/.EasyOCR."
        ) from exc


def _read_number(reader, mask: np.ndarray) -> int | None:
    """Recognise one all-digit number in ``mask``, or ``None``.

    ``None`` covers every way this frame can fail to produce a number: no
    text found (a destroyed tower or an undamaged king shows none), a
    non-digit result, or a confidence below ``TOWER_HP_MIN_CONFIDENCE``.
    The caller keeps the last known HP rather than acting on a guess.

    The crop's own box is passed as ``horizontal_list`` so EasyOCR runs
    only its recogniser; there is nothing to detect when calibration
    already says where the digits are.
    """
    height, width = mask.shape[:2]
    results = reader.recognize(
        mask,
        horizontal_list=[[0, width, 0, height]],
        free_list=[],
        allowlist=TOWER_HP_ALLOWLIST,
        detail=1,
    )
    digits = "".join(str(word).strip() for _box, word, _conf in results)
    if not digits.isdigit():
        return None
    if min(conf for _box, _word, conf in results) < TOWER_HP_MIN_CONFIDENCE:
        return None
    return int(digits)


class TowerHealthReader:
    """Reads tower HP for the six towers with EasyOCR.

    Each region is read independently: one recogniser call per crop, with
    the crop's own bounding box handed in so EasyOCR's text *detector*
    never runs. Detection is pure waste here because calibration already
    says where the digits are, and skipping it costs nothing in accuracy
    (11/12 either way) while removing any chance of one tower's reading
    being attributed to another. Batching the six into one call measured
    the same ~84 ms, so the simpler form wins.

    A reading is returned only if it is all digits and scores at least
    ``TOWER_HP_MIN_CONFIDENCE``; anything else is ``None``, meaning "this
    frame could not be read", which the caller treats as keep-last-known.

    Owns one lazily built EasyOCR reader, released by :meth:`close`. NOT
    thread-safe: read from one thread only, or give each thread its own.
    """

    def __init__(self) -> None:
        self._reader: object | None = None

    def read(self, frame: np.ndarray) -> dict[str, int | None]:
        reader = self._get_reader()
        out: dict[str, int | None] = {k: None for k in TOWER_KEYS}
        for key, region in TOWER_HP_REGIONS.items():
            mask = _glyph_mask(
                _crop_region(frame, region, f"TOWER_HP_REGIONS[{key!r}]",
                             margin=_TOWER_CROP_MARGIN_PX)
            )
            out[key] = _read_number(reader, mask)
        return out

    def close(self) -> None:
        """Drop the EasyOCR reader. Idempotent; a later :meth:`read`
        transparently builds a fresh one."""
        self._reader = None

    def __enter__(self) -> TowerHealthReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----- internals -----

    def _get_reader(self):
        if self._reader is None:
            self._reader = _make_easyocr_reader()
        return self._reader


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
