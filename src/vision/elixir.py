"""Read the current elixir count from the number beside the elixir bar.

The game draws the current elixir as an integer at the left end of the
elixir bar, in a fixed position at a fixed size. That makes this an
11-way classification (0-10) rather than an OCR problem, so it is solved
by matching against 11 reference crops instead of by a recogniser:
matching is exact, costs microseconds, and is structurally incapable of
returning 7 for a 1 the way Tesseract did for the tower digits.

Why read it at all, when ``GameState`` already simulates elixir from the
in-game regeneration rates: the simulation drifts. It has no ground truth
for what the match actually gave you - a spell that refunds, a mirror, a
placement the game rejected after ``spend_elixir`` already debited it -
and the error accumulates over a 5-minute match with nothing to correct
it. The number on screen is the authority; see
``GameState.set_elixir``.

The display shows the FLOOR of the true value, which is the conservative
direction for affordability: a displayed 4 means at least 4, so a 4-cost
card really is playable. It can never be optimistic, so acting on it
cannot produce a rejected placement.

Reference crops live in ``src/assets/templates/elixir/`` as ``0.png`` ..
``10.png`` and are captured by ``python -m src.main --calibrate elixir``.
They are stored as raw BGR crops rather than as pre-computed masks, so
the comparison mask is re-derived from the current ``hud`` constants on
both sides and cannot go stale if those are ever retuned - and so the
files stay human-inspectable when a reading looks wrong.

``ELIXIR_DIGIT_REGION`` is CALIBRATE, like every other per-machine pixel
constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .hud import crop_region, glyph_bool

# (x_frac, y_frac, w_frac, h_frac) of the elixir count, drawn at the left
# end of the elixir bar along the bottom of the arena view. CALIBRATE FOR
# YOUR RESOLUTION - this is a starting guess only, positioned just left of
# ``lifecycle.ELIXIR_BAR_PATCH``. The box must be wide enough to hold the
# two-digit "10", since a crop that clips it makes 10 unreadable at
# exactly the moment elixir matters most.
ELIXIR_DIGIT_REGION: tuple[float, float, float, float] = (0.2824, 0.9501, 0.0378, 0.0323)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "assets" / "templates" / "elixir"

# Highest value the counter can show. The bar caps at 10.
MAX_ELIXIR_READING = 10

# Every template and every live crop is resized to this (width, height)
# before comparison, so a template captured at one window size still
# matches at another.
SIGNATURE_SIZE = (48, 32)

# A match is believed only if the best template overlaps this well AND
# beats the runner-up by this margin. The margin is what makes an
# occluded crop - a card dragged over the counter, an elixir-collector
# animation - read as "unreadable" instead of as whichever digit it
# happens to resemble most. Same principle as the tower readers'
# confidence gate: abstaining costs one cycle, a wrong value corrupts the
# elixir model until the next good read.
MIN_SIGNATURE_IOU = 0.60
MIN_SIGNATURE_IOU_MARGIN = 0.08


def _signature(crop: np.ndarray) -> np.ndarray:
    """Canonical-size boolean glyph mask used for comparison.

    The glyph is cut to its own bounding box, scaled to fit
    ``SIGNATURE_SIZE`` preserving aspect ratio, and centred there. That
    makes the signature independent of *where* in the crop the digits
    landed, which matters because the crop origin is derived from the
    BlueStacks window position on every grab: nudging the window one pixel
    shifts every crop one pixel. Compared without this normalisation, a
    single pixel of shift cost ~25% of readings (they abstained rather
    than misreading, but abstaining means falling back to the simulated
    elixir this reader exists to correct).

    Aspect ratio is preserved rather than stretched away because it is
    what most cleanly separates the two-digit "10" from a lone "1".

    An empty mask - counter occluded, or a crop with no glyph in it -
    yields an all-False signature, which scores 0 against everything and
    is rejected by the caller's IoU gate.
    """
    mask = glyph_bool(crop)
    ys, xs = np.nonzero(mask)
    canvas = np.zeros((SIGNATURE_SIZE[1], SIGNATURE_SIZE[0]), dtype=bool)
    if ys.size == 0:
        return canvas

    glyph = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    gh, gw = glyph.shape
    target_w, target_h = SIGNATURE_SIZE
    scale = min(target_w / gw, target_h / gh)
    new_w = max(1, min(target_w, int(round(gw * scale))))
    new_h = max(1, min(target_h, int(round(gh * scale))))
    scaled = cv2.resize(
        glyph.astype(np.uint8) * 255, (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    ) >= 128

    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = scaled
    return canvas


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean glyph masks.

    IoU rather than a plain pixel-agreement rate because background
    dominates these crops: two different digits already agree on ~80% of
    pixels just by both being mostly background, which leaves no usable
    margin between the best and second-best match.
    """
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / union


# Records which ELIXIR_DIGIT_REGION the templates on disk were captured
# against. Re-calibrating the region invalidates every one of them - the
# crop is framed differently, so nothing matches and every frame reads as
# unreadable. That failure is invisible without this file: the reader
# simply stops returning values and the elixir silently reverts to pure
# simulation, which is the bug this module exists to fix.
REGION_STAMP_PATH = TEMPLATE_DIR / "region.json"

# Fractions are printed to 4 dp by the calibration wizard, so that is the
# precision at which a stamp and the committed constant can agree.
_REGION_DECIMALS = 4


def _rounded(region: tuple[float, float, float, float]) -> list[float]:
    return [round(float(v), _REGION_DECIMALS) for v in region]


def template_path(value: int) -> Path:
    return TEMPLATE_DIR / f"{value}.png"


def save_template(
    frame: np.ndarray,
    value: int,
    region: tuple[float, float, float, float] = ELIXIR_DIGIT_REGION,
) -> Path:
    """Write the current elixir crop as the reference for ``value``.

    Used by the ``elixir`` calibration phase. Stores the raw crop; the
    comparison mask is derived at load time.

    ``region`` is passed explicitly because calibration captures templates
    with the box the user *just drew*, which is not yet the committed
    ``ELIXIR_DIGIT_REGION`` - it is still sitting in the wizard's output
    waiting to be pasted. It is stamped alongside the crops so a later
    mismatch is caught rather than silently breaking every read.
    """
    if not 0 <= value <= MAX_ELIXIR_READING:
        raise ValueError(
            f"Elixir template value must be 0..{MAX_ELIXIR_READING}, got {value}"
        )
    crop = crop_region(frame, region, "ELIXIR_DIGIT_REGION")
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = template_path(value)
    if not cv2.imwrite(str(path), crop):
        raise RuntimeError(f"Failed to write elixir template to {path}")
    REGION_STAMP_PATH.write_text(
        json.dumps({"region": _rounded(region)}), encoding="utf-8"
    )
    return path


def _check_region_stamp() -> None:
    """Raise if the templates on disk were captured against a different
    region than the one currently committed."""
    if not REGION_STAMP_PATH.is_file():
        raise RuntimeError(
            f"Elixir templates in {TEMPLATE_DIR} carry no {REGION_STAMP_PATH.name} "
            "stamp, so there is no way to tell whether they were captured "
            "against the committed ELIXIR_DIGIT_REGION. Re-capture them with "
            "'python -m src.main --calibrate elixir'."
        )
    stamped = json.loads(REGION_STAMP_PATH.read_text(encoding="utf-8"))["region"]
    current = _rounded(ELIXIR_DIGIT_REGION)
    if stamped != current:
        raise RuntimeError(
            f"Elixir templates were captured against region {stamped} but "
            f"ELIXIR_DIGIT_REGION is now {current}. The crops are framed "
            "differently, so every frame would read as unreadable. If you "
            "just re-ran 'python -m src.main --calibrate elixir', paste the "
            "ELIXIR_DIGIT_REGION it printed into src/vision/elixir.py; "
            "otherwise re-run it to re-capture the crops."
        )


def missing_template_values() -> list[int]:
    """Which of 0..10 have no reference crop on disk yet."""
    return [
        v for v in range(MAX_ELIXIR_READING + 1)
        if not template_path(v).is_file()
    ]


class ElixirReader:
    """Classifies the on-screen elixir count against reference crops.

    :meth:`read` returns the displayed integer, or ``None`` when this
    frame could not be classified confidently - counter occluded, between
    animation frames, or no template captured for the value on screen.
    ``None`` is an ordinary per-frame outcome: the caller keeps simulating
    until the next good read.

    Templates are loaded lazily on first read and held as canonical
    signatures. A directory with no templates at all raises rather than
    silently never reading anything; a partial set is reported once and
    then works for the values it has, because capturing all eleven takes
    a full match and a half-calibrated setup should still be usable.
    """

    def __init__(self) -> None:
        self._signatures: dict[int, np.ndarray] | None = None

    def read(self, frame: np.ndarray) -> int | None:
        signatures = self._get_signatures()
        crop = crop_region(frame, ELIXIR_DIGIT_REGION, "ELIXIR_DIGIT_REGION")
        scores = self._score(crop, signatures)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (best_value, best_score) = ranked[0]
        if best_score < MIN_SIGNATURE_IOU:
            return None
        if len(ranked) > 1 and best_score - ranked[1][1] < MIN_SIGNATURE_IOU_MARGIN:
            return None
        return best_value

    def score_table(self, frame: np.ndarray) -> dict[int, float]:
        """IoU of the current crop against every loaded template.

        Exposed for calibration and debugging: the separation between the
        best and second-best score is the only evidence that
        ``MIN_SIGNATURE_IOU_MARGIN`` is set anywhere sensible.
        """
        crop = crop_region(frame, ELIXIR_DIGIT_REGION, "ELIXIR_DIGIT_REGION")
        return self._score(crop, self._get_signatures())

    def close(self) -> None:
        """Drop the loaded templates. Idempotent; a later :meth:`read`
        reloads them from disk."""
        self._signatures = None

    def __enter__(self) -> ElixirReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----- internals -----

    @staticmethod
    def _score(
        crop: np.ndarray, signatures: dict[int, np.ndarray]
    ) -> dict[int, float]:
        live = _signature(crop)
        return {value: _iou(live, sig) for value, sig in signatures.items()}

    def _get_signatures(self) -> dict[int, np.ndarray]:
        if self._signatures is not None:
            return self._signatures

        signatures: dict[int, np.ndarray] = {}
        for value in range(MAX_ELIXIR_READING + 1):
            path = template_path(value)
            if not path.is_file():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(
                    f"Elixir template {path} exists but could not be decoded; "
                    "delete it and re-capture with "
                    "'python -m src.main --calibrate elixir'"
                )
            signatures[value] = _signature(image)

        if not signatures:
            raise RuntimeError(
                f"No elixir reference crops in {TEMPLATE_DIR}. Elixir is read "
                "from screen rather than simulated, so this is required: "
                "capture them with 'python -m src.main --calibrate elixir' "
                "during a live match."
            )
        _check_region_stamp()

        missing = [v for v in range(MAX_ELIXIR_READING + 1) if v not in signatures]
        if missing:
            print(
                f"Elixir templates missing for {missing} - those values will "
                f"read as unreadable and fall back to the simulated elixir. "
                f"Re-run 'python -m src.main --calibrate elixir' to add them."
            )
        self._signatures = signatures
        return signatures
