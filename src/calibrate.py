"""Interactive calibration wizard for per-machine pixel coordinates.

Usage::

    python -m src.main --calibrate           # every phase
    python -m src.main --calibrate timer     # just the match timer
    # or directly:
    python -m src.calibrate [PHASE]

Six phases, run in order by default (follow the on-screen prompts;
rectangles are two clicks, top-left then bottom-right):
  1. ``viewport`` — mark the game viewport rectangle on the raw window.
  2. ``hand``     — click the centre of each of the four hand-card slots.
  3. ``towers``   — mark each tower's HP-BAR rectangle (6 total). The bar,
     not the number: HP is read as the bar's fill fraction.
  4. ``timer``    — mark the "m:ss" match countdown rectangle.
  5. ``crowns``   — mark the two crown rows on the POSTMATCH screen, used
     to read the final crown score. Needs the postmatch screen up, so it
     is the one phase that does NOT want a live match.
  6. ``elixir``   — mark the elixir count, then capture one labelled
     reference crop per value (0-10) by keypress. This phase is the only
     one that both prints a constant AND writes files:
     ``src/assets/templates/elixir/``.

Any single phase can be run on its own by name, so re-calibrating one
constant does not mean redoing the others. Only the constants for the
phases actually run are printed.

Phases 3, 4 and 6 read in-match HUD elements, so they need a live match on
screen. Phase 5 (``crowns``) is the exception: it needs the POSTMATCH
screen instead. Phases 1 and 2 need neither (the hand is visible in-match
only, but its slots do not move, so the menu is fine for phase 1).

Phases 2-6 report fractions of the *cropped* game viewport, not of the raw
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
PHASE_NAMES: tuple[str, ...] = (
    "viewport", "hand", "towers", "timer", "crowns", "elixir",
)
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


def _capture_elixir_templates(
    window_title: str, region: tuple[float, float, float, float]
) -> None:
    """Capture one labelled reference crop per elixir value, live.

    The elixir counter is classified against 11 reference crops rather
    than read by OCR, so those crops have to come from this machine's
    pixels once. Values are labelled by keypress: ``0``-``9`` for those
    values, ``a`` for 10, ``q`` or Esc to finish. The crop being labelled
    is the one on screen in the preview, so there is no race between
    reading the number and pressing the key.

    The preview is moved to the top-left of the screen because
    :meth:`ScreenCapture.grab` re-reads the window rectangle on every
    grab: a preview window sitting over the game viewport gets captured
    instead of the game. That failure is self-announcing - the preview
    fills with a recursive image of itself - and dragging the window off
    the viewport fixes it.
    """
    from .vision.elixir import (
        MAX_ELIXIR_READING,
        missing_template_values,
        save_template,
    )
    from .vision.hud import crop_region

    key_for = {str(v): v for v in range(10)}
    key_for["a"] = 10

    print(
        f"\nPhase 5 (elixir): capture one reference crop per elixir value"
        f" — needs a LIVE MATCH on screen\n"
        f"  Keys: 0-9 = that value, a = 10, q/Esc = done.\n"
        f"  Play normally and label the number as it changes; you need all"
        f" of 0..{MAX_ELIXIR_READING}.\n"
        f"  If the preview shows a picture of itself, drag it off the game"
        f" viewport — it is being captured instead of the game."
    )

    with ScreenCapture(window_title) as cap:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.moveWindow(_WIN, 0, 0)
        cv2.resizeWindow(_WIN, 520, 260)
        while True:
            frame = _grab_painted(cap.grab, window_title)
            crop = crop_region(frame, region, "ELIXIR_DIGIT_REGION")
            preview = cv2.resize(
                crop, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_NEAREST
            )
            canvas = np.zeros((preview.shape[0] + 70, max(preview.shape[1], 500), 3),
                              dtype=np.uint8)
            canvas[:preview.shape[0], :preview.shape[1]] = preview
            missing = missing_template_values()
            cv2.putText(canvas, f"still needed: {missing or 'none - press q'}",
                        (8, preview.shape[0] + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
            cv2.putText(canvas, "0-9 = value   a = 10   q = done",
                        (8, preview.shape[0] + 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
            cv2.imshow(_WIN, canvas)

            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            value = key_for.get(chr(key)) if 32 <= key < 127 else None
            if value is not None:
                path = save_template(frame, value, region)
                print(f"  saved elixir={value} -> {path}")

    remaining = missing_template_values()
    if remaining:
        print(
            f"  WARNING: no crop captured for {remaining}. Those values will "
            f"read as unreadable and fall back to simulated elixir; re-run "
            f"'--calibrate elixir' to add them."
        )
    else:
        print(f"  All 0..{MAX_ELIXIR_READING} captured.")


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

    if cropped is None and any(
        p in ("hand", "towers", "timer", "elixir", "crowns") for p in phases
    ):
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
    elixir_region: tuple[float, float, float, float] | None = None
    crown_regions: dict[str, tuple[float, float, float, float]] | None = None

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
            # Phase 3: tower HP BARS — not the HP numbers. Tower HP is now
            # read as the bar's fill fraction, so the box must bound the bar
            # itself, tightly: the fraction is relative to the box, so a box
            # wider than the bar reads a full tower as less than full.
            print("\nPhase 3 (towers): drag a box around each tower's HP BAR"
                  " — the coloured strip, NOT the number"
                  " — needs a LIVE MATCH on screen")
            print("  Bound the bar tightly left-to-right: fill is measured as a"
                  " fraction of the box width, so extra width reads as missing HP.")
            tower_keys = [
                "enemy_king", "enemy_left", "enemy_right",
                "friendly_king", "friendly_left", "friendly_right",
            ]
            tower_regions = {}
            for key in tower_keys:
                r_tl, r_br = _collect_rect(cropped, f"HP BAR: {key}")
                tower_regions[key] = _fractions(r_tl, r_br, cw, ch)

        if "timer" in phases:
            # Phase 4: match timer — drag a box around the "m:ss" countdown
            print("\nPhase 4 (timer): drag a box around the match timer (m:ss)"
                  " — needs a LIVE MATCH on screen, the timer only exists in-match")
            t_tl, t_br = _collect_rect(
                cropped, "Match timer region (m:ss, in-match only)"
            )
            timer_region = _fractions(t_tl, t_br, cw, ch)

        if "crowns" in phases:
            # Phase 5: the two postmatch crown rows. Needs the POSTMATCH
            # screen on display, not a live match. The blue/friendly side is
            # always the bottom row, so the side is fixed per region; boxing
            # them in the wrong order would transpose the score.
            print("\nPhase 6 (crowns): drag a box around each row of crown"
                  " slots on the POSTMATCH screen (top row, then bottom)")
            print("  Box each row snugly: a crown must clear a fraction of the"
                  " box area, so an oversized box can hide real crowns.")
            crown_regions = {}
            for side, where in (("enemy", "TOP"), ("friendly", "BOTTOM")):
                c_tl, c_br = _collect_rect(
                    cropped,
                    f"Crown row: {side} ({where} row, all 3 slots, snug)",
                )
                crown_regions[side] = _fractions(c_tl, c_br, cw, ch)

        if "elixir" in phases:
            # Phase 5a: elixir counter region — the box the reference crops
            # are captured through, so it is drawn before they are taken.
            print("\nPhase 5 (elixir): drag a box around the elixir COUNT"
                  " (the number beside the elixir bar)"
                  " — needs a LIVE MATCH on screen")
            e_tl, e_br = _collect_rect(
                cropped, "Elixir count region (include room for a 2-digit 10)"
            )
            elixir_region = _fractions(e_tl, e_br, cw, ch)

    cv2.destroyAllWindows()

    if elixir_region is not None:
        # Phase 5b: the labelled reference crops, taken through the box just
        # drawn rather than the committed constant — which is still the old
        # value until the user pastes what this run prints.
        _capture_elixir_templates(window_title, elixir_region)
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

    if tower_regions is not None:
        print("# src/vision/towers.py")
        print("TOWER_BAR_REGIONS: dict[str, tuple[float, float, float, float]] = {")
        for key, (xf, yf, wf, hf) in tower_regions.items():
            print(f'    "{key}":  ({xf:.4f}, {yf:.4f}, {wf:.4f}, {hf:.4f}),')
        print("}")
        print()
    if timer_region is not None:
        print("# src/vision/ocr.py")
        xf, yf, wf, hf = timer_region
        print(
            "MATCH_TIMER_REGION: tuple[float, float, float, float] = "
            f"({xf:.4f}, {yf:.4f}, {wf:.4f}, {hf:.4f})"
        )

    if crown_regions is not None:
        print("# src/vision/crowns.py")
        print("CROWN_ROW_REGIONS: dict[str, tuple[float, float, float, float]] = {")
        for key, (xf, yf, wf, hf) in crown_regions.items():
            print(f'    "{key}": ({xf:.4f}, {yf:.4f}, {wf:.4f}, {hf:.4f}),')
        print("}")
        print()

    if elixir_region is not None:
        xf, yf, wf, hf = elixir_region
        print("\n# src/vision/elixir.py")
        print(
            "ELIXIR_DIGIT_REGION: tuple[float, float, float, float] = "
            f"({xf:.4f}, {yf:.4f}, {wf:.4f}, {hf:.4f})"
        )
        print(
            "# ^ REQUIRED: the reference crops just captured are stamped with "
            "this region\n#   and the reader refuses to use them until the "
            "constant matches."
        )


if __name__ == "__main__":
    import sys

    from .main import WINDOW_TITLE
    calibrate(WINDOW_TITLE, sys.argv[1] if len(sys.argv) > 1 else ALL_PHASES)
