"""Tracks elapsed match time, elixir, tower HP, and match outcome.

Match time runs off a local clock, but that clock is *anchored* to the
on-screen ``m:ss`` countdown: ``start_match`` can only stamp the moment
the elixir bar became visible, which is several seconds before the real
3-2-1 countdown ends, so ``anchor_match_clock`` re-derives
``match_start_time`` from the first trustworthy timer reading
(``MatchTimerReader``). Without it the clock runs permanently ~5s ahead.

Elixir is READ from screen and only interpolated by simulation. The
regeneration rates are reproduced exactly (2.8s, 1.4s, 0.93s per pip
across normal/double/triple phases) and ``spend_elixir`` debits a
committed placement, but simulation alone has no ground truth for what
the match actually granted and drifts over five minutes with nothing to
correct it. ``set_elixir`` pins the value to the integer the HUD shows
each perception cycle; the simulation's job is reduced to carrying the
fraction between reads, since the display only shows the floor.

Tower HP and match result are externally driven: ``TowerBarReader`` and
``MatchLifecycle`` push values in via ``update_tower_hp`` and
``set_match_result``.

Tower HP is NORMALISED, 0.0-1.0, and is the fill fraction of the
on-screen HP bar rather than an HP number over a learned maximum. That
deletes the max-learning this class used to do, and with it the failure
where one 5147 misread inflated a denominator for a whole match.

DESTRUCTION is still inferred rather than read, because the game draws no
bar at all over rubble - so there is no empty bar to see, and a bar hidden
behind a fight looks identical to a destroyed one in any single frame.
What separates them is duration, so destruction requires a sustained run
of absences and stays provisional: a tower that reads again retracts it,
crown included. Princess towers only; a king kill is the banner's verdict,
not vision's.

Destroyed towers award crowns but never decide the match. ``match_result``
comes from ``MatchLifecycle`` reading the victory/defeat banner, because
tower vision is accurate enough to shape reward and not accurate enough to
end an episode.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

MatchResult = Literal["win", "loss", "draw"]

# Tower HP is now NORMALISED (0.0-1.0), read as the fill fraction of the
# on-screen HP bar rather than as an HP number divided by a learned maximum.
# A full tower is 1.0 by construction, so there is no per-account or
# per-opponent tower level to account for and no maximum to infer.
FULL_TOWER_HP = 1.0

# Consecutive unreadable frames before a tower counts as destroyed. Absence is
# still the only destruction signal: the game draws no bar over rubble, so the
# reader returns None rather than 0.0, and a bar hidden behind a fight is
# indistinguishable from a destroyed one in any single frame. What separates
# them is duration.
#
# This was 30 (~7.5s) when the signal was OCR on the HP digits, which read
# only ~36-54% of frames and so needed a long run before absence meant
# anything. Bar detection reads a *visible* bar essentially every frame, so
# the same confidence takes far less evidence: 8 frames is ~2s at a 0.25s
# cycle. Nothing downstream needs destruction promptly - it shapes reward and
# never ends an episode - and a premature call is retracted by
# TOWER_READS_TO_UNDO_DESTRUCTION, so this trades latency for a bounded error.
TOWER_MISSING_READS_FOR_DESTROYED = 8

# Successful readings before a tower may be considered destroyed at all. One
# reading is not enough: a single spurious read on a king's crop - which holds
# the gold level badge before the king is damaged - was enough to make the
# king's ORDINARY silence look like destruction, ending a 67s match as a false
# "win". A live tower draws its number steadily and clears this in about a
# second, while a misread is isolated by definition.
TOWER_READS_BEFORE_DESTRUCTIBLE = 3

# Consecutive readings that RETRACT a destruction. Destruction is inferred from
# absence, and absence is what an ordinary fight over a tower produces, so it
# is a hypothesis rather than a fact - and rubble never produces a number, so a
# tower that reads again was never destroyed. Retracting costs nothing and
# bounds how much damage a premature call can do: the crown is handed back and
# the HP-delta reward the false zero paid out is refunded by the same
# arithmetic that granted it.
#
# Two readings rather than one so a single misread cannot resurrect a tower
# that really is rubble.
TOWER_READS_TO_UNDO_DESTRUCTION = 2

# Towers whose destruction may be inferred from absence. Princess towers only.
#
# Kings stay excluded, but for a different reason than before. Under OCR a
# king's HP number was almost never legible (0-29% of frames against ~80% for
# princess towers), so absence said nothing about it - and trusting it anyway
# declared enemy_king destroyed at 112s in a match that ran to 191s, with the
# crown count reaching an impossible 3-3. Bar detection removes that specific
# problem: a king's bar reads as reliably as a princess tower's.
#
# They remain excluded because a king kill ENDS the match, and that verdict
# comes from ``MatchLifecycle`` reading the victory/defeat banner - the same
# screen a human looks at. A vision-derived king death can only either agree
# with the banner, in which case it added nothing, or contradict it, in which
# case it is wrong. The cost is that a genuinely destroyed king keeps its last
# reading for the handful of frames before the banner lands: a cosmetic error
# in an episode that is already over, against false 3-crown wins mid-match.
ABSENCE_DESTRUCTIBLE_TOWERS: tuple[str, ...] = (
    "friendly_left", "friendly_right", "enemy_left", "enemy_right",
)

# Normalised HP a tower is assumed to have before its bar is first read.
DEFAULT_TOWER_HP: dict[str, float] = {
    "friendly_left": FULL_TOWER_HP,
    "friendly_right": FULL_TOWER_HP,
    "friendly_king": FULL_TOWER_HP,
    "enemy_left": FULL_TOWER_HP,
    "enemy_right": FULL_TOWER_HP,
    "enemy_king": FULL_TOWER_HP,
}

# Below this normalised HP the enemy king counts as damaged, and therefore
# activated. Full-health bars measured exactly 1.000 on real pixels, so this
# only has to sit clear of measurement noise, not of a distribution.
KING_ACTIVATION_HP = 0.98

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

    # Largest gap between an on-screen timer reading and the running clock
    # that is still accepted as a correction. A genuine correction is the
    # few seconds between the elixir bar appearing and the match actually
    # starting. A big gap means the reading is not regulation time at all -
    # the overtime countdown restarts from 2:00 and would anchor the clock
    # two minutes early - or it is a plain OCR misread. Reject either.
    ANCHOR_MAX_CORRECTION_SEC = 30.0

    def __init__(self):
        self.match_start_time: Optional[float] = None
        self.clock_anchored: bool = False
        self.current_elixir: float = self.STARTING_ELIXIR
        self.last_elixir_update: Optional[float] = None
        self.is_match_active: bool = False

        # Normalised 0.0-1.0, read as HP-bar fill. None never appears here:
        # an unreadable frame keeps the last known value.
        self.tower_hp: dict[str, float] = dict(DEFAULT_TOWER_HP)

        self.crowns_friendly: int = 0
        self.crowns_enemy: int = 0
        self.match_result: Optional[MatchResult] = None
        self._destroyed_towers: set[str] = set()
        # How many times each tower's HP has been read this match. Absence of a
        # reading only means "destroyed" for a tower that was reliably readable
        # before: an undamaged king shows no number at all, so it never
        # accumulates reads and can never be mistaken for destroyed.
        self._read_counts: dict[str, int] = {k: 0 for k in TOWER_KEYS}
        self._missing_reads: dict[str, int] = {k: 0 for k in TOWER_KEYS}
        # Consecutive readings seen for a tower currently believed destroyed,
        # which is the evidence that belief was wrong.
        self._revive_reads: dict[str, int] = {k: 0 for k in TOWER_KEYS}

    # ----- lifecycle -----

    def start_match(self) -> None:
        """Begin a match with a provisional clock.

        ``match_start_time`` is only a guess until :meth:`anchor_match_clock`
        replaces it with a value derived from the on-screen timer: the
        caller detects the match from the elixir bar, which is already
        visible during the pre-match countdown.
        """
        now = time.time()
        self.match_start_time = now
        self.clock_anchored = False
        self.last_elixir_update = now
        self.current_elixir = self.STARTING_ELIXIR
        self.is_match_active = True
        self.match_result = None
        self.crowns_friendly = 0
        self.crowns_enemy = 0
        for k in TOWER_KEYS:
            self.tower_hp[k] = DEFAULT_TOWER_HP[k]
            self._missing_reads[k] = 0
            self._revive_reads[k] = 0
        self._destroyed_towers.clear()
        for k in TOWER_KEYS:
            self._read_counts[k] = 0
        print("Match started!")

    def end_match(self, result: Optional[MatchResult] = None) -> None:
        if result is not None:
            self.match_result = result
        self.is_match_active = False
        print(f"Match ended! result={self.match_result}")

    # ----- timing -----

    def anchor_match_clock(self, remaining_sec: float) -> bool:
        """Re-derive ``match_start_time`` from an on-screen timer reading.

        ``remaining_sec`` is seconds left on the displayed countdown (see
        ``MatchTimerReader``), so the implied elapsed time is
        ``REGULAR_TIME_END - remaining_sec``. Returns True if the reading
        was adopted as ground truth, False if it was rejected - in which
        case the caller should try the next frame.

        Only the first accepted reading moves the clock. Local time and
        game time advance at the same rate once they agree, so one anchor
        suffices, and re-anchoring every frame would instead make
        ``match_time`` jitter by up to a second on the timer's own
        whole-second quantization (the display shows 2:59 for the whole
        interval where 179.0-179.999s remain, so a single anchor is
        accurate to ~1s and biased ~0.5s late).

        Readings that are rejected:

        - a full ``REGULAR_TIME_END`` (3:00), which is also what the game
          shows frozen during the pre-match countdown and therefore cannot
          be told apart from a match that has genuinely just begun;
        - anything outside 0..REGULAR_TIME_END, which neither countdown
          can display;
        - anything implying an elapsed time more than
          ``ANCHOR_MAX_CORRECTION_SEC`` from the current clock: an
          overtime countdown or an OCR misread, not a correction.

        ``last_elixir_update`` is deliberately untouched. Elixir accrual is
        measured from that stamp, so moving it into the future would make
        ``update_elixir`` compute negative gain and silently drain elixir.
        """
        if not self.is_match_active or self.match_start_time is None:
            return False
        if self.clock_anchored:
            return False
        if not 0.0 <= remaining_sec < self.REGULAR_TIME_END:
            return False
        implied_elapsed = self.REGULAR_TIME_END - remaining_sec
        drift = implied_elapsed - self.get_elapsed_seconds()
        if abs(drift) > self.ANCHOR_MAX_CORRECTION_SEC:
            return False
        self.match_start_time = time.time() - implied_elapsed
        self.clock_anchored = True
        print(
            f"Match clock anchored to on-screen timer "
            f"({remaining_sec:.0f}s left, elapsed {implied_elapsed:.1f}s, "
            f"corrected by {drift:+.1f}s)"
        )
        return True

    def get_elapsed_seconds(self) -> float:
        """Seconds since match start, floored at 0 and NOT capped.

        Use this for "has this match run too long" decisions:
        :meth:`get_current_match_time` saturates at ``MATCH_MAX_DURATION``
        and so can never exceed a timeout set above it.
        """
        if not self.is_match_active or self.match_start_time is None:
            return 0.0
        return max(0.0, time.time() - self.match_start_time)

    def get_current_match_time(self) -> float:
        """Elapsed match seconds, clamped to ``MATCH_MAX_DURATION``.

        The clamp is load-bearing for the observation's ``time_norm``
        (normalized by the same constant, so it stays in 0..1) and for the
        phase/elixir-rate lookups. Timeouts want :meth:`get_elapsed_seconds`.
        """
        return min(self.get_elapsed_seconds(), self.MATCH_MAX_DURATION)

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
        # Floored at 0: regeneration must never run backwards. A negative
        # interval (a clock adjustment, or a future-dated stamp) would
        # otherwise subtract elixir here with nothing spent.
        elapsed = max(0.0, now - self.last_elixir_update)
        gained = elapsed / self.get_elixir_rate()
        self.current_elixir = min(self.MAX_ELIXIR, self.current_elixir + gained)
        self.last_elixir_update = now

    def get_current_elixir(self) -> float:
        self.update_elixir()
        return self.current_elixir

    def set_elixir(self, value: Optional[int]) -> None:
        """Adopt an on-screen elixir reading as ground truth.

        ``value`` is the integer the HUD shows (see ``ElixirReader``), or
        ``None`` for a frame that could not be classified - in which case
        the simulation carries the value to the next good read.

        The display is the FLOOR of the true elixir, so a reading of 4 is
        consistent with anything in [4, 5). When the simulated value
        already falls inside that window it is KEPT, which is what
        preserves the sub-pip fraction the digit cannot show. Otherwise the
        simulation has drifted and is snapped back onto the reading.

        Snapping to the floor rather than to the middle of the window is
        deliberate: it can only ever understate how much elixir we have, so
        a placement this value says is affordable really is. The reverse
        error produces a rejected placement and a -0.05 reward for
        something the policy did nothing wrong to earn.

        ``last_elixir_update`` is restamped either way, so the accrual
        measured by :meth:`update_elixir` starts from this reading rather
        than double-counting the interval before it.
        """
        if value is None:
            return
        if not 0 <= value <= int(self.MAX_ELIXIR):
            raise ValueError(
                f"Elixir reading {value} is outside 0..{int(self.MAX_ELIXIR)}; "
                "the reader classifies against templates for those values "
                "only, so this means its output was corrupted"
            )
        floor = float(value)
        if not floor <= self.current_elixir < floor + 1.0:
            self.current_elixir = min(self.MAX_ELIXIR, floor)
        self.last_elixir_update = time.time()

    def spend_elixir(self, amount: float) -> bool:
        self.update_elixir()
        if self.current_elixir + 1e-6 >= amount:
            self.current_elixir -= amount
            return True
        return False

    # ----- towers -----

    def set_tower_hp(self, key: str, value: Optional[float]) -> None:
        """Absorb one normalised HP reading, or ``None`` for no visible bar.

        ``value`` is the tower's HP-bar fill fraction, 0.0-1.0 (see
        ``TowerBarReader``). ``None`` means no bar was visible in this
        frame, which is a destroyed tower AND a bar hidden behind a fight -
        the two are indistinguishable in any single frame, so neither is
        acted on immediately.

        Destruction is therefore inferred from a *run* of absences, never
        from a single one. Only ``ABSENCE_DESTRUCTIBLE_TOWERS`` take part,
        and only after ``TOWER_READS_BEFORE_DESTRUCTIBLE`` successful reads,
        so a tower whose bar was never located by calibration cannot be
        destroyed by its own permanent silence.

        A destroyed tower keeps being watched rather than being written off:
        rubble never grows a bar, so any later reading disproves the
        destruction. See :meth:`_retract_tower_destroyed`.
        """
        if key not in self.tower_hp:
            raise KeyError(f"Unknown tower key {key!r}; expected one of {TOWER_KEYS}")
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Tower HP for {key!r} must be a normalised 0.0-1.0 bar fill, "
                f"got {value!r}. Raw HP points are no longer used; see "
                "src/vision/towers.py"
            )
        if value is not None and value <= 0.0:
            # No bar colour found. Same meaning as None: not a reading of
            # zero HP, because the game draws no empty bar over rubble.
            value = None
        if key in self._destroyed_towers:
            # Keep watching it. Rubble never grows a bar, so a reading here
            # disproves the destruction rather than updating it.
            if value is None:
                self._revive_reads[key] = 0
                return
            self._revive_reads[key] += 1
            if self._revive_reads[key] < TOWER_READS_TO_UNDO_DESTRUCTION:
                return
            self._retract_tower_destroyed(key)
            # Fall through: this is now an ordinary reading for a live tower.
        if value is None:
            if (
                self.is_match_active
                and key in ABSENCE_DESTRUCTIBLE_TOWERS
                and self._read_counts[key] >= TOWER_READS_BEFORE_DESTRUCTIBLE
            ):
                self._missing_reads[key] += 1
                if self._missing_reads[key] >= TOWER_MISSING_READS_FOR_DESTROYED:
                    self.tower_hp[key] = 0.0
                    self._register_tower_destroyed(key)
            return  # otherwise keep the last known HP
        self._missing_reads[key] = 0
        self._read_counts[key] += 1
        self.tower_hp[key] = float(value)

    def update_tower_hp(self, readings: dict[str, Optional[float]]) -> None:
        for k, v in readings.items():
            self.set_tower_hp(k, v)

    def _register_tower_destroyed(self, key: str) -> None:
        """Record a destroyed tower and award the crown for it.

        Deliberately does NOT decide the match. Vision-derived tower state is
        good enough to shape reward but not to end an episode: a single bad
        frame used to be able to declare a win, overruling a lifecycle layer
        that was still correctly reporting IN_MATCH. ``match_result`` now
        comes only from ``MatchLifecycle``, which reads the victory/defeat
        banner - the same screen a human would look at.
        """
        if key in self._destroyed_towers:
            return
        self._destroyed_towers.add(key)
        self._revive_reads[key] = 0
        if key.startswith("enemy"):
            self.crowns_friendly = min(3, self.crowns_friendly + 1)
        else:
            self.crowns_enemy = min(3, self.crowns_enemy + 1)

    def _retract_tower_destroyed(self, key: str) -> None:
        """Undo a destruction a later reading disproved.

        Hands the crown back and clears the counters so the tower is tracked
        normally again - and can still be destroyed later, for real. Loud on
        purpose: a retraction means absence-based destruction fired early, and
        that is worth seeing in a run's output rather than inferring from a
        recording afterwards.
        """
        self._destroyed_towers.discard(key)
        self._revive_reads[key] = 0
        self._missing_reads[key] = 0
        if key.startswith("enemy"):
            self.crowns_friendly = max(0, self.crowns_friendly - 1)
        else:
            self.crowns_enemy = max(0, self.crowns_enemy - 1)
        print(f"Tower {key} read again after being marked destroyed - "
              f"retracting (crowns now {self.crowns_friendly}-{self.crowns_enemy})")

    def is_enemy_left_alive(self) -> bool:
        return "enemy_left" not in self._destroyed_towers

    def is_enemy_right_alive(self) -> bool:
        return "enemy_right" not in self._destroyed_towers

    def is_enemy_king_active(self) -> bool:
        """Whether the enemy king tower is firing.

        Two independent signals, matching the game's own rules: the king
        activates when it takes damage, and it also activates when either
        enemy princess tower falls.

        The damage test is now a direct comparison against full health,
        which normalised bar fill makes possible. Under OCR it was not: max
        HP was learned from the readings themselves and the king's first
        reading was already post-damage, so current and max were equal
        exactly when it had just activated - the test had to fall back to
        "have we ever managed to read this king's number at all".
        """
        return (
            self.tower_hp["enemy_king"] < KING_ACTIVATION_HP
            or bool({"enemy_left", "enemy_right"} & self._destroyed_towers)
        )

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
            f"Elixir: {self.current_elixir:.1f}/{self.MAX_ELIXIR:.0f} | "
            f"Phase: {self.get_match_phase()} | "
            f"Rate: {self.get_elixir_rate():.2f}s/elixir | "
            f"Crowns: {self.crowns_friendly}-{self.crowns_enemy}"
        )
