"""OCR-based tower-HP reader.

Reads small regions of the captured frame around each of the six towers
and runs Tesseract on the cropped numbers. Coordinates are stored as
fractions of the captured monitor (0-1) so they survive resolution
changes - but the regions themselves still need to be calibrated for
your BlueStacks crop. See ``TOWER_HP_REGIONS`` below.

Pytesseract is imported lazily so the rest of the backend works even
when Tesseract isn't installed; ``read`` will return all-``None``
readings in that case.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .gameState import TOWER_KEYS

try:
    import pytesseract  # type: ignore
    _TESS_AVAILABLE = True
except Exception:
    pytesseract = None  # type: ignore
    _TESS_AVAILABLE = False


# Each entry is (x_frac, y_frac, w_frac, h_frac) within the captured frame.
# CALIBRATE FOR YOUR RESOLUTION: the values below are reasonable defaults
# for a portrait BlueStacks crop and almost certainly need adjustment.
TOWER_HP_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "friendly_left":  (0.05, 0.62, 0.12, 0.05),
    "friendly_right": (0.83, 0.62, 0.12, 0.05),
    "friendly_king":  (0.44, 0.78, 0.12, 0.05),
    "enemy_left":     (0.05, 0.30, 0.12, 0.05),
    "enemy_right":    (0.83, 0.30, 0.12, 0.05),
    "enemy_king":     (0.44, 0.18, 0.12, 0.05),
}


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    # Upscale + threshold makes Tesseract substantially more reliable on
    # the small UI digits.
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


class TowerHealthReader:
    def __init__(self, regions: Optional[dict] = None, tesseract_cmd: Optional[str] = None):
        self.regions = regions or TOWER_HP_REGIONS
        if tesseract_cmd and _TESS_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._available = _TESS_AVAILABLE

    @property
    def available(self) -> bool:
        return self._available

    def read(self, frame: np.ndarray) -> dict[str, Optional[int]]:
        out: dict[str, Optional[int]] = {k: None for k in TOWER_KEYS}
        if not self._available or frame is None or frame.size == 0:
            return out

        h, w = frame.shape[:2]
        for key, (xf, yf, wf, hf) in self.regions.items():
            x0 = max(0, int(xf * w))
            y0 = max(0, int(yf * h))
            x1 = min(w, x0 + max(1, int(wf * w)))
            y1 = min(h, y0 + max(1, int(hf * h)))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            try:
                processed = _preprocess_for_ocr(crop)
                text = pytesseract.image_to_string(
                    processed,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789",
                ).strip()
                if text:
                    out[key] = int(text)
            except Exception:
                out[key] = None
        return out
