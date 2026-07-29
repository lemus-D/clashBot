"""Interactive calibration wizard for per-machine pixel coordinates.

Usage::

    python -m src.main --calibrate           # all four phases
    python -m src.main --calibrate timer     # just the match timer
    # or directly:
    python -m src.calibrate [PHASE]

Four phases, run in order by default (follow the on-screen prompts;
rectangles are two clicks, top-left then bottom-right):
  1. ``viewport`` — mark the game viewport rectangle on the raw window.
  2. ``hand``     — click the centre of each of the four hand-card slots.
  3. ``towers``   — mark each tower's HP-number rectangle (6 total).
  4. ``timer``    — mark the "m:ss" match countdown rectangle.

Any single phase can be run on its own by name, so re-calibrating one
constant does not mean redoing the others. Only the constants for the
phases actually run are printed.

Phases 3 and 4 read in-match HUD elements, so they need a live match on
screen; phases 1 and 2 do not (the hand is visible in-match only, but its
slots do not move, so the menu is fine for phase 1).

Phases 2-4 report fractions of the *cropped* game viewport, not of the raw
window. When phase 1 runs, that crop is the rectangle just drawn; when it
is skipped, the frame comes from :class:`ScreenCapture` with the committed
``WINDOW_CROP_*`` constants — i.e. byte-for-byte the frame the runtime
feeds the OCR readers, which is what makes the fractions valid to paste.

At the end the new constants are printed; paste them into the source files shown.
"""

from __future__ import annotations

import time
from typing import Callable

import cv2
import mss
import numpy as np
import pywinctl as gw

from .vision.capture import (
    WINDOW_CROP_BOTTOM,
    WINDOW_CROP_LEFT,
    WINDOW_CROP_RIGHT,
    WINDOW_CROP_TOP,
    ScreenCapture,
)


_WIN = "clashBot calibration"

# Phase names accepted by :func:`calibrate`, in run order. "all" is also
# accepted and means every one of them.
PHASE_NAMES: tuple[str, ...] = ("viewport", "hand", "towers", "timer")
ALL_PHASES = "all"


def _grab_painted(
    grab: Callable[[], np.ndarray],
    window_title: str,
    timeout: float = 3.0,
    poll: float = 0.15,
) -> np.ndarray:
    """Call ``grab`` until it returns a frame that isn't solid black.

    BlueStacks renders via a hardware-accelerated surface, so the first
    frame or two right after activate() can come back solid black before
    the GPU has actually painted - poll until a real frame shows up.
    """
    deadline = time.time() + timeout
    while True:
        img = grab()
        if img.mean() > 2.0:
            return img
        if time.time() >= deadline:
            raise RuntimeError(
                f"Captured frame from {window_title!r} is black after "
                f"{timeout:.1f}s. The window is likely minimized, occluded "
                "by another window, or BlueStacks is rendering through a "
                "hardware overlay that screen capture can't see. Make sure "
                "the BlueStacks window is fully visible and focused, then "
                "try switching its graphics renderer (BlueStacks Settings "
                "> Display > Graphics engine, e.g. DirectX <-> OpenGL) or "
                "disabling 'Hardware-accelerated GPU scheduling' in "
                "Windows Graphics settings."
            )
        time.sleep(poll)


def _grab_raw(window_title: str) -> tuple[np.ndarray, int, int]:
    """Uncropped screenshot of the whole window, plus its size."""
    wins = gw.getWindowsWithTitle(window_title)
    if not wins:
        raise RuntimeError(f"Window not found: {window_title!r}")
    w = wins[0]
    if w.isMinimized:
        w.restore()
    w.activate()
    time.sleep(0.5)
    mon = {"top": w.top, "left": w.left, "width": w.width, "height": w.height}

    with mss.mss() as sct:
        frame = _grab_painted(
            lambda: cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR),
            window_title,
        )
    return frame, w.width, w.height


def _grab_cropped(window_title: str) -> np.ndarray:
    """Cropped game viewport, taken through the runtime capture path.

    Used when phase 1 is skipped. Fractions reported by phases 2-4 are
    relative to the viewport, so this must be the *same* frame the runtime
    sees - hence ``ScreenCapture`` with its committed ``WINDOW_CROP_*``
    defaults rather than a crop re-derived here. (Phase 1's own
    ``frame_raw[top:bottom, left:right]`` is exactly what
    ``ScreenCapture.monitor`` describes, so the two paths agree by
    construction.)
    """
    with ScreenCapture(window_title) as cap:
        return _grab_painted(cap.grab, window_title)


def _collect_points(frame: np.ndarray, prompts: list[str]) -> list[tuple[int, int]]:
    """Show frame; collect one left-click per prompt. Returns pixel coords."""
    pts: list[tuple[int, int]] = []

    def on_mouse(evt, x, y, flags, _):
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WIN, on_mouse)

    while len(pts) < len(prompts):
        img = frame.copy()
        cv2.putText(img, prompts[len(pts)], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(img, p, 6, (0, 255, 0), -1)
        cv2.imshow(_WIN, img)
        cv2.waitKey(30)

    return pts


def _collect_rect(frame: np.ndarray, label: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Collect a rectangle via two clicks (TL then BR) with rubber-band preview."""
    pts: list[tuple[int, int]] = []
    mouse = [(-1, -1)]

    def on_mouse(evt, x, y, flags, _):
        mouse[0] = (x, y)
        if evt == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WIN, on_mouse)

    while len(pts) < 2:
        img = frame.copy()
        prompt = f"{label}  —  click top-left" if not pts else f"{label}  —  click bottom-right"
        cv2.putText(img, prompt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if pts:
            cv2.circle(img, pts[0], 4, (0, 255, 0), -1)
            cv2.rectangle(img, pts[0], mouse[0], (0, 255, 0), 1)
        cv2.imshow(_WIN, img)
        cv2.waitKey(30)

    return pts[0], pts[1]


def resolve_phases(phase: str) -> tuple[str, ...]:
    """Phases to run for a ``--calibrate`` value. ``"all"`` means every one.

    Raises ``ValueError`` on an unknown name rather than falling back to
    running everything - a typo must not silently make the user redo all
    four phases.
    """
    if phase == ALL_PHASES:
        return PHASE_NAMES
    if phase not in PHASE_NAMES:
        raise ValueError(
            f"Unknown calibration phase {phase!r}; valid phases are "
            f"{', '.join(PHASE_NAMES)}, or {ALL_PHASES!r} / no value at all "
            "to run every phase"
        )
    return (phase,)


def _fractions(
    tl: tuple[int, int], br: tuple[int, int], width: int, height: int
) -> tuple[float, float, float, float]:
    """A clicked rectangle as (x_frac, y_frac, w_frac, h_frac) of the frame."""
    return (
        tl[0] / width,
        tl[1] / height,
        (br[0] - tl[0]) / width,
        (br[1] - tl[1]) / height,
    )


def calibrate(window_title: str, phase: str = ALL_PHASES) -> None:
    """Run the calibration wizard, or just one ``phase`` of it.

    ``phase`` is ``"all"`` (every phase, the default) or one name from
    :data:`PHASE_NAMES`. Only the constants for the phases run are printed.
    """
    phases = resolve_phases(phase)
    print("Calibration starting — follow the on-screen prompts. Press nothing; just click.")
    print(f"Phases to run: {', '.join(phases)}")

    crop: tuple[int, int, int, int] | None = None       # top, left, right, bottom
    cropped: np.ndarray | None = None

    if "viewport" in phases:
        # Phase 1: crop — define the game viewport on the raw window screenshot
        print("\nPhase 1 (viewport): drag the game viewport boundary")
        frame_raw, win_w, win_h = _grab_raw(window_title)
        tl, br = _collect_rect(frame_raw, "Game viewport")
        crop = (tl[1], tl[0], win_w - br[0], win_h - br[1])
        # Cropped frame used for all remaining phases. Identical to what
        # ScreenCapture produces from the constants above, so a run that
        # skips this phase can substitute _grab_cropped().
        cropped = frame_raw[tl[1]:br[1], tl[0]:br[0]]

    if cropped is None and any(p in ("hand", "towers", "timer") for p in phases):
        # Phase 1 was skipped: the remaining phases still need the viewport,
        # so take it the way the runtime does.
        print(
            f"\nViewport from the committed crop in src/vision/capture.py "
            f"(top={WINDOW_CROP_TOP}, left={WINDOW_CROP_LEFT}, "
            f"right={WINDOW_CROP_RIGHT}, bottom={WINDOW_CROP_BOTTOM}). "
            "Re-run with 'viewport' if those are stale."
        )
        cropped = _grab_cropped(window_title)

    card_fracs: list[tuple[float, float]] | None = None
    tower_regions: dict[str, tuple[float, float, float, float]] | None = None
    timer_region: tuple[float, float, float, float] | None = None

    if cropped is not None:
        ch, cw = cropped.shape[:2]

        if "hand" in phases:
            # Phase 2: card slots — click the centre of each of the four hand slots
            print("\nPhase 2 (hand): click the centre of each hand-card slot (left to right)")
            card_pts = _collect_points(
                cropped,
                [f"Card slot {i + 1}  —  click centre" for i in range(4)],
            )
            card_fracs = [(x / cw, y / ch) for x, y in card_pts]

        if "towers" in phases:
            # Phase 3: tower HP regions — drag a box around each HP number
            print("\nPhase 3 (towers): drag a box around each tower's HP number"
                  " — needs a LIVE MATCH on screen")
            tower_keys = [
                "enemy_king", "enemy_left", "enemy_right",
                "friendly_king", "friendly_left", "friendly_right",
            ]
            tower_regions = {}
            for key in tower_keys:
                r_tl, r_br = _collect_rect(cropped, f"HP region: {key}")
                tower_regions[key] = _fractions(r_tl, r_br, cw, ch)

        if "timer" in phases:
            # Phase 4: match timer — drag a box around the "m:ss" countdown
            print("\nPhase 4 (timer): drag a box around the match timer (m:ss)"
                  " — needs a LIVE MATCH on screen, the timer only exists in-match")
            t_tl, t_br = _collect_rect(
                cropped, "Match timer region (m:ss, in-match only)"
            )
            timer_region = _fractions(t_tl, t_br, cw, ch)

    cv2.destroyAllWindows()

    # Print results — only the constants for the phases that ran
    print("\n=== Calibration complete — paste these into your source files ===\n")

    if crop is not None:
        crop_top, crop_left, crop_right, crop_bottom = crop
        print("# src/vision/capture.py")
        print(f"WINDOW_CROP_TOP    = {crop_top}")
        print(f"WINDOW_CROP_LEFT   = {crop_left}")
        print(f"WINDOW_CROP_RIGHT  = {crop_right}")
        print(f"WINDOW_CROP_BOTTOM = {crop_bottom}")
        print()

    if card_fracs is not None:
        print("# src/env/actions.py")
        print("HAND_CARD_POSITIONS: tuple[tuple[float, float], ...] = (")
        for fx, fy in card_fracs:
            print(f"    ({fx:.4f}, {fy:.4f}),")
        print(")")
        print()

    if tower_regions is not None or timer_region is not None:
        print("# src/vision/ocr.py")
    if tower_regions is not None:
        print("TOWER_HP_REGIONS: dict[str, tuple[float, float, float, float]] = {")
        for key, (xf, yf, wf, hf) in tower_regions.items():
            print(f'    "{key}":  ({xf:.4f}, {yf:.4f}, {wf:.4f}, {hf:.4f}),')
        print("}")
    if timer_region is not None:
        xf, yf, wf, hf = timer_region
        print(
            "MATCH_TIMER_REGION: tuple[float, float, float, float] = "
            f"({xf:.4f}, {yf:.4f}, {wf:.4f}, {hf:.4f})"
        )


if __name__ == "__main__":
    import sys

    from .main import WINDOW_TITLE
    calibrate(WINDOW_TITLE, sys.argv[1] if len(sys.argv) > 1 else ALL_PHASES)
