"""Tracks elapsed match time, elixir, tower HP, and match outcome.

Elixir regeneration is simulated rather than read from screen - it
reproduces the in-game rates exactly (2.8s, 1.4s, 0.93s per pip across
normal/double/triple phases) and is corrected by ``spend_elixir`` when
the action layer commits a placement.

Tower HP and match result are externally driven: ``TowerHealthReader``
and ``MatchLifecycle`` push values in via ``update_tower_hp`` and
``set_match_result``.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

MatchResult = Literal["win", "loss", "draw"]

DEFAULT_PRINCESS_HP = 2534
DEFAULT_KING_HP = 4824

DEFAULT_TOWER_HP: dict[str, int] = {
    "friendly_left": DEFAULT_PRINCESS_HP,
    "friendly_right": DEFAULT_PRINCESS_HP,
    "friendly_king": DEFAULT_KING_HP,
    "enemy_left": DEFAULT_PRINCESS_HP,
    "enemy_right": DEFAULT_PRINCESS_HP,
    "enemy_king": DEFAULT_KING_HP,
}

TOWER_KEYS: tuple[str, ...] = tuple(DEFAULT_TOWER_HP)


class GameState:
    """Match-time + elixir + tower-HP bookkeeping.

    Phases handled automatically: normal (0:00-2:00), double (2:00-3:00),
    overtime double (3:00-4:00), overtime triple (4:00-5:00).
    """

    NORMAL_ELIXIR_RATE = 2.8
    DOUBLE_ELIXIR_RATE = 1.4
    TRIPLE_ELIXIR_RATE = 0.93

    DOUBLE_ELIXIR_START = 120
    REGULAR_TIME_END = 180
    TRIPLE_ELIXIR_START = 240
    MATCH_MAX_DURATION = 300

    STARTING_ELIXIR = 5.0
    MAX_ELIXIR = 10.0

    def __init__(self):
        self.match_start_time: Optional[float] = None
        self.current_elixir: float = self.STARTING_ELIXIR
        self.last_elixir_update: Optional[float] = None
        self.is_match_active: bool = False

        self.tower_hp: dict[str, Optional[int]] = dict(DEFAULT_TOWER_HP)
        self.tower_max_hp: dict[str, int] = dict(DEFAULT_TOWER_HP)

        self.crowns_friendly: int = 0
        self.crowns_enemy: int = 0
        self.match_result: Optional[MatchResult] = None
        self._destroyed_towers: set[str] = set()

    # ----- lifecycle -----

    def start_match(self) -> None:
        now = time.time()
        self.match_start_time = now
        self.last_elixir_update = now
        self.current_elixir = self.STARTING_ELIXIR
        self.is_match_active = True
        self.match_result = None
        self.crowns_friendly = 0
        self.crowns_enemy = 0
        for k in TOWER_KEYS:
            self.tower_hp[k] = self.tower_max_hp[k]
        self._destroyed_towers.clear()
        print("Match started!")

    def end_match(self, result: Optional[MatchResult] = None) -> None:
        if result is not None:
            self.match_result = result
        self.is_match_active = False
        print(f"Match ended! result={self.match_result}")

    # ----- timing -----

    def get_current_match_time(self) -> float:
        if not self.is_match_active or self.match_start_time is None:
            return 0
        elapsed = time.time() - self.match_start_time
        return min(elapsed, self.MATCH_MAX_DURATION)

    def get_match_phase(self) -> str:
        if not self.is_match_active:
            return "ended"
        elapsed = self.get_current_match_time()
        if elapsed < self.DOUBLE_ELIXIR_START:
            return "normal"
        if elapsed < self.REGULAR_TIME_END:
            return "double"
        if elapsed < self.TRIPLE_ELIXIR_START:
            return "overtime_double"
        return "overtime_triple"

    # ----- elixir -----

    def get_elixir_rate(self) -> float:
        elapsed = self.get_current_match_time()
        if elapsed < self.DOUBLE_ELIXIR_START:
            return self.NORMAL_ELIXIR_RATE
        if elapsed < self.TRIPLE_ELIXIR_START:
            return self.DOUBLE_ELIXIR_RATE
        return self.TRIPLE_ELIXIR_RATE

    def update_elixir(self) -> None:
        if not self.is_match_active:
            return
        now = time.time()
        if self.last_elixir_update is None:
            self.last_elixir_update = now
            return
        elapsed = now - self.last_elixir_update
        gained = elapsed / self.get_elixir_rate()
        self.current_elixir = min(self.MAX_ELIXIR, self.current_elixir + gained)
        self.last_elixir_update = now

    def get_current_elixir(self) -> float:
        self.update_elixir()
        return self.current_elixir

    def spend_elixir(self, amount: float) -> bool:
        self.update_elixir()
        if self.current_elixir + 1e-6 >= amount:
            self.current_elixir -= amount
            return True
        return False

    # ----- towers -----

    def set_tower_hp(self, key: str, value: Optional[int]) -> None:
        if key not in self.tower_hp:
            raise KeyError(f"Unknown tower key {key!r}; expected one of {TOWER_KEYS}")
        if value is None:
            return  # OCR couldn't read this frame; keep the last known HP
        self.tower_hp[key] = value
        if value <= 0:
            self._register_tower_destroyed(key)

    def update_tower_hp(self, readings: dict[str, Optional[int]]) -> None:
        for k, v in readings.items():
            self.set_tower_hp(k, v)

    def _register_tower_destroyed(self, key: str) -> None:
        if key in self._destroyed_towers:
            return
        self._destroyed_towers.add(key)
        if key.startswith("enemy"):
            self.crowns_friendly = min(3, self.crowns_friendly + 1)
            if key == "enemy_king":
                self.set_match_result("win")
        else:
            self.crowns_enemy = min(3, self.crowns_enemy + 1)
            if key == "friendly_king":
                self.set_match_result("loss")

    def is_enemy_left_alive(self) -> bool:
        v = self.tower_hp["enemy_left"]
        return v is None or v > 0

    def is_enemy_right_alive(self) -> bool:
        v = self.tower_hp["enemy_right"]
        return v is None or v > 0

    def is_enemy_king_active(self) -> bool:
        v = self.tower_hp["enemy_king"]
        return v is not None and v < self.tower_max_hp["enemy_king"]

    # ----- result -----

    def set_match_result(self, result: MatchResult) -> None:
        self.match_result = result

    # ----- formatting -----

    def get_formatted_time(self) -> str:
        elapsed = self.get_current_match_time()
        return f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"

    def get_status_string(self) -> str:
        if not self.is_match_active:
            return "No active match"
        return (
            f"Time: {self.get_formatted_time()} | "
            f"Elixir: {self.current_elixir:.1f}/10 | "
            f"Phase: {self.get_match_phase()} | "
            f"Rate: {self.get_elixir_rate():.2f}s/elixir | "
            f"Crowns: {self.crowns_friendly}-{self.crowns_enemy}"
        )
