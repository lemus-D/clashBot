"""Interactive calibration wizard for per-machine pixel coordinates.

Usage::

    python -m src.main --calibrate
    # or directly:
    python -m src.calibrate

Three phases (follow the on-screen prompts; rectangles are two clicks,
top-left then bottom-right):
  1. Crop    — mark the game viewport rectangle on the raw window.
  2. Cards   — click the centre of each of the four hand-card slots.
  3. Towers  — mark each tower's HP-number rectangle (6 total).

At the end the new constants are printed; paste them into the source files shown.
"""

from __future__ import annotations

import time

import cv2
import mss
import numpy as np
import pywinctl as gw


_WIN = "clashBot calibration"


def _grab_raw(window_title: str, timeout: float = 3.0, poll: float = 0.15) -> tuple[np.ndarray, int, int]:
    wins = gw.getWindowsWithTitle(window_title)
    if not wins:
        raise RuntimeError(f"Window not found: {window_title!r}")
    w = wins[0]
    if w.isMinimized:
        w.restore()
    w.activate()
    time.sleep(0.5)
    mon = {"top": w.top, "left": w.left, "width": w.width, "height": w.height}

    # BlueStacks renders via a hardware-accelerated surface, so the first
    # frame or two right after activate() can come back solid black before
    # the GPU has actually painted - poll until a real frame shows up.
    deadline = time.time() + timeout
    with mss.mss() as sct:
        while True:
            img = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
            if img.mean() > 2.0:
                return img, w.width, w.height
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


def calibrate(window_title: str) -> None:
    print("Calibration starting — follow the on-screen prompts. Press nothing; just click.")

    frame_raw, win_w, win_h = _grab_raw(window_title)

    # Phase 1: crop — define the game viewport on the raw window screenshot
    print("\nPhase 1: drag the game viewport boundary")
    tl, br = _collect_rect(frame_raw, "Game viewport")
    crop_left, crop_top = tl
    crop_right = win_w - br[0]
    crop_bottom = win_h - br[1]

    # Cropped frame used for all remaining phases
    cropped = frame_raw[tl[1]:br[1], tl[0]:br[0]]
    ch, cw = cropped.shape[:2]

    # Phase 2: card slots — click the centre of each of the four hand slots
    print("\nPhase 2: click the centre of each hand-card slot (left to right)")
    card_pts = _collect_points(
        cropped,
        [f"Card slot {i + 1}  —  click centre" for i in range(4)],
    )
    card_fracs = [(x / cw, y / ch) for x, y in card_pts]

    # Phase 3: tower HP regions — drag a box around each HP number
    print("\nPhase 3: drag a box around each tower's HP number")
    tower_keys = [
        "enemy_king", "enemy_left", "enemy_right",
        "friendly_king", "friendly_left", "friendly_right",
    ]
    tower_regions: dict[str, tuple[float, float, float, float]] = {}
    for key in tower_keys:
        r_tl, r_br = _collect_rect(cropped, f"HP region: {key}")
        tower_regions[key] = (
            r_tl[0] / cw,
            r_tl[1] / ch,
            (r_br[0] - r_tl[0]) / cw,
            (r_br[1] - r_tl[1]) / ch,
        )

    cv2.destroyAllWindows()

    # Print results
    print("\n=== Calibration complete — paste these into your source files ===\n")

    print("# src/vision/capture.py")
    print(f"WINDOW_CROP_TOP    = {crop_top}")
    print(f"WINDOW_CROP_LEFT   = {crop_left}")
    print(f"WINDOW_CROP_RIGHT  = {crop_right}")
    print(f"WINDOW_CROP_BOTTOM = {crop_bottom}")

    print()
    print("# src/env/actions.py")
    print("HAND_CARD_POSITIONS: tuple[tuple[float, float], ...] = (")
    for fx, fy in card_fracs:
        print(f"    ({fx:.4f}, {fy:.4f}),")
    print(")")

    print()
    print("# src/vision/ocr.py")
    print("TOWER_HP_REGIONS: dict[str, tuple[float, float, float, float]] = {")
    for key, (xf, yf, wf, hf) in tower_regions.items():
        print(f'    "{key}":  ({xf:.4f}, {yf:.4f}, {wf:.4f}, {hf:.4f}),')
    print("}")


if __name__ == "__main__":
    from .main import WINDOW_TITLE
    calibrate(WINDOW_TITLE)
