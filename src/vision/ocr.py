"""OCR-based tower-HP reader.

Crops a small region around each of the six towers and runs Tesseract
on the digits. Regions are fractions of the captured frame (0-1);
CALIBRATE ``TOWER_HP_REGIONS`` for your BlueStacks crop.

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

Tesseract is a hard requirement: a missing install, missing tessdata, a
malformed crop, or an OCR failure all raise rather than degrading silently.
"""

from __future__ import annotations

import os
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


# PSM.SINGLE_BLOCK (psm 6) = "a uniform block of text": Tesseract segments
# the stack into one line per crop. Tower HP is always digits.
OCR_PSM = PSM.SINGLE_BLOCK
OCR_WHITELIST = "0123456789"

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


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Binarize one tower-HP crop to black digits on a white background.

    Polarity is decided from this crop alone, never from its neighbours, so
    that stacking crops cannot make one region's contents change another's
    rendering. Orientation comes from the pixel counts: digits occupy a
    small minority of one of these tight HP boxes - 12-22% of pixels across
    the six TOWER_HP_REGIONS on synthetic renders - so the majority class
    after Otsu is background and is forced to white. Should some region ever
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
        crops = [_preprocess_for_ocr(self._crop(frame, k)) for k in keys]
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
        for text, (left, top, _right, bottom) in self._words(api):
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
            tessdata = _resolve_tessdata_dir()
            try:
                api = PyTessBaseAPI(path=tessdata, lang="eng", psm=OCR_PSM)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Tesseract failed to initialise from tessdata {tessdata!r}: "
                    f"{exc}. Tower-HP OCR is required; check the Tesseract "
                    "install and that eng.traineddata matches its version."
                ) from exc
            api.SetVariable("tessedit_char_whitelist", OCR_WHITELIST)
            self._api = api
        return self._api

    @staticmethod
    def _words(api: PyTessBaseAPI) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Recognised words as (text, bounding box) pairs.

        An all-blank composite - every tower destroyed or obscured, which is
        legitimate - leaves the iterator empty, and tesserocr raises
        "No text returned" if that is read anyway. Skip empty elements so the
        caller gets six Nones instead of an exception.
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

    @staticmethod
    def _crop(frame: np.ndarray, key: str) -> np.ndarray:
        h, w = frame.shape[:2]
        xf, yf, wf, hf = TOWER_HP_REGIONS[key]
        x0 = max(0, int(xf * w))
        y0 = max(0, int(yf * h))
        x1 = min(w, x0 + max(1, int(wf * w)))
        y1 = min(h, y0 + max(1, int(hf * h)))
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            raise ValueError(
                f"Tower HP region for {key!r} produced an empty crop "
                f"({x0},{y0})-({x1},{y1}) from a {w}x{h} frame; "
                "check TOWER_HP_REGIONS calibration"
            )
        return crop
