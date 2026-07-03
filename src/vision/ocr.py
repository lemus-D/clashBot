"""OCR-based tower-HP reader.

Crops a small region around each of the six towers and runs Tesseract
on the digits. Regions are fractions of the captured frame (0-1);
CALIBRATE ``TOWER_HP_REGIONS`` for your BlueStacks crop.

Documented exception to fail-loud: pytesseract is optional. When it is
missing (or an individual OCR call flakes out), ``read`` returns
``None`` for the affected towers instead of raising.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..game.state import TOWER_KEYS

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore


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


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Upscale + threshold makes Tesseract substantially more reliable on
    # the small UI digits.
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


class TowerHealthReader:
    """Reads tower HP for the six towers via Tesseract OCR."""

    def read(self, frame: np.ndarray) -> dict[str, int | None]:
        out: dict[str, int | None] = {k: None for k in TOWER_KEYS}
        if pytesseract is None or frame is None or frame.size == 0:
            return out

        h, w = frame.shape[:2]
        for key, (xf, yf, wf, hf) in TOWER_HP_REGIONS.items():
            x0 = max(0, int(xf * w))
            y0 = max(0, int(yf * h))
            x1 = min(w, x0 + max(1, int(wf * w)))
            y1 = min(h, y0 + max(1, int(hf * h)))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            processed = _preprocess_for_ocr(crop)
            try:
                text = pytesseract.image_to_string(
                    processed,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789",
                ).strip()
            except Exception:
                continue  # documented exception: OCR flakiness degrades to None
            if text.isdigit():
                out[key] = int(text)
        return out
