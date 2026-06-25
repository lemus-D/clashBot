"""Screen capture wrapper around ``mss`` + ``pywinctl``.

The window crop offsets are the BlueStacks-specific magic numbers that
used to live inside ``windowCap.start_window_cap``. They strip away the
side toolbars and top chrome so the captured frame is just the game
viewport. CALIBRATE these for your BlueStacks resolution and theme.
"""

from __future__ import annotations

import time
from typing import Optional

import mss
import numpy as np
import pywinctl as gw
import cv2


WINDOW_CROP_TOP = 50
WINDOW_CROP_LEFT = 383
WINDOW_CROP_RIGHT = 50
WINDOW_CROP_BOTTOM = 0


class ScreenCapture:
    """Context-managed ``mss`` capture targeting a named window.

    Usage::

        with ScreenCapture("BlueStacks App Player 1") as cap:
            frame = cap.grab()
            monitor = cap.monitor
    """

    def __init__(
        self,
        window_title: str,
        crop_top: int = WINDOW_CROP_TOP,
        crop_left: int = WINDOW_CROP_LEFT,
        crop_right: int = WINDOW_CROP_RIGHT,
        crop_bottom: int = WINDOW_CROP_BOTTOM,
        activate: bool = True,
    ):
        self.window_title = window_title
        self.crop_top = crop_top
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_bottom = crop_bottom
        self.activate_on_open = activate

        self._sct: Optional[mss.base.MSSBase] = None
        self._window = None

    # ----- context management -----

    def __enter__(self) -> "ScreenCapture":
        self._sct = mss.mss().__enter__()
        try:
            self._window = gw.getWindowsWithTitle(self.window_title)[0]
        except IndexError as exc:
            raise RuntimeError(
                f"No window found with title: {self.window_title}"
            ) from exc
        if self.activate_on_open:
            try:
                if self._window.isMinimized:
                    self._window.restore()
                self._window.activate()
                time.sleep(0.5)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._sct is not None:
            self._sct.__exit__(exc_type, exc, tb)
            self._sct = None

    # ----- frame access -----

    @property
    def monitor(self) -> dict:
        if self._window is None:
            raise RuntimeError("ScreenCapture is not open")
        return {
            "top": self._window.top + self.crop_top,
            "left": self._window.left + self.crop_left,
            "width": self._window.width - self.crop_left - self.crop_right,
            "height": self._window.height - self.crop_top - self.crop_bottom,
        }

    def grab(self) -> np.ndarray:
        if self._sct is None:
            raise RuntimeError("ScreenCapture is not open")
        shot = self._sct.grab(self.monitor)
        img = np.array(shot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
