"""Tracks elapsed match time, elixir, tower HP, and match outcome.

Match time runs off a local clock, but that clock is *anchored* to the
on-screen ``m:ss`` countdown: ``start_match`` can only stamp the moment
the elixir bar became visible, which is several seconds before the real
3-2-1 countdown ends, so ``anchor_match_clock`` re-derives
``match_start_time`` from the first trustworthy timer reading
(``MatchTimerReader``). Without it the clock runs permanently ~5s ahead.

Elixir regeneration is simulated rather than read from screen - it
reproduces the in-game rates exactly (2.8s, 1.4s, 0.93s per pip across
normal/double/triple phases) and is corrected by ``spend_elixir`` when
the action layer commits a placement.

Tower HP and match result are externally driven: ``TowerHealthReader``
and ``MatchLifecycle`` push values in via ``update_tower_hp`` and
``set_match_result``.

Two things about tower HP are inferred rather than read. Each tower's
MAXIMUM is learned as the largest HP seen this match, because HP only
falls and because tower level - and so max HP - varies per account and
per opponent. DESTRUCTION is inferred from a run of unreadable frames,
because a destroyed tower keeps neither bar nor number and so has no zero
to read. Princess towers only: their number is legible most frames, so a
long absence is evidence, whereas a king's is legible almost never even
after it is damaged, so absence says nothing about it. Destruction is
provisional - a tower that reads again afterwards retracts it, crown
included, since rubble never shows a number.

Destroyed towers award crowns but never decide the match. ``match_result``
comes from ``MatchLifecycle`` reading the victory/defeat banner, because
tower vision is accurate enough to shape reward and not accurate enough to
end an episode.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

MatchResult = Literal["win", "loss", "draw"]

# Starting guesses only - see ``set_tower_hp``, which raises a tower's max to
# the largest HP it actually reads during the match. Tower HP depends on tower
# LEVEL, and the level differs per account and per opponent, so no constant is
# right for the enemy side: these were 2534/4824 (level ~11 tournament values)
# against an account whose princess towers read 1512, which normalised a
# full-health tower to 0.60. Observed on this account: princess (level 2) 1512,
# friendly king (level 3) 2736, a level-2 enemy king 2532.
DEFAULT_PRINCESS_HP = 1512
DEFAULT_KING_HP = 2736

# Consecutive unreadable frames before a tower counts as destroyed. Absence is
# the only destruction signal available: the game draws neither bar nor number
# over rubble, and the reader returns None rather than 0 (it never sees a 0 to
# read). At a ~0.25s cycle this is ~7.5s.
#
# It was 10 (~2.5s) and that was too little evidence: a match destroyed two
# live princess towers 0.3s apart when a fight covered both numbers at once.
# Both really did die later, which is what made the mistake easy to miss.
# Nothing downstream needs destruction promptly - it shapes reward and never
# ends an episode - so the cost of waiting is far below the cost of a false
# positive.
TOWER_MISSING_READS_FOR_DESTROYED = 30

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

# Towers whose destruction may be inferred from absence. Princess towers only,
# and this is a property of the vision rather than a preference. Measured over
# one match by dumping frames and checking them against the recording, the read
# rate while a tower was alive was:
#
#   enemy_left 83%   friendly_right 80%   enemy_right 80%
#   friendly_king 29%   enemy_king 0%
#
# At ~80% a long run of misses really does mean the number is gone. A king's
# number is only intermittently legible even after it is damaged, so absence
# says nothing about it: one match declared enemy_king destroyed at 112s and a
# dump read it alive at 1810 HP sixty seconds later, while the match ran on to
# 191s - a real king kill would have ended it on the spot. Both kings "died"
# at unrelated times and the crown count reached an impossible 3-3.
#
# The cost of excluding them is that a genuinely destroyed king keeps its last
# reading instead of dropping to 0 for the last frames before the banner ends
# the episode. That is a cosmetic error in an episode that is already over,
# against false 3-crown wins in the middle of live matches.
ABSENCE_DESTRUCTIBLE_TOWERS: tuple[str, ...] = (
    "friendly_left", "friendly_right", "enemy_left", "enemy_right",
)

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

        self.tower_hp: dict[str, Optional[int]] = dict(DEFAULT_TOWER_HP)
        self.tower_max_hp: dict[str, int] = dict(DEFAULT_TOWER_HP)

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
        # Max HP is re-learned every match: the opponent changes, and with them
        # the enemy towers' level. Carrying a previous opponent's higher max
        # over would normalise this match's full-health towers below 1.0.
        self.tower_max_hp = dict(DEFAULT_TOWER_HP)
        for k in TOWER_KEYS:
            self.tower_hp[k] = self.tower_max_hp[k]
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

    def spend_elixir(self, amount: float) -> bool:
        self.update_elixir()
        if self.current_elixir + 1e-6 >= amount:
            self.current_elixir -= amount
            return True
        return False

    # ----- towers -----

    def set_tower_hp(self, key: str, value: Optional[int]) -> None:
        """Absorb one HP reading, or ``None`` for an unreadable frame.

        Destruction is inferred from a *run* of unreadable frames rather
        than from a zero reading, because the game never draws a zero: a
        destroyed tower loses its bar and its number entirely. The old
        ``value <= 0`` trigger could therefore only ever fire on a misread,
        and did - one bogus ``enemy_king`` zero ended a match at 2:28 with a
        false "win". Only ``ABSENCE_DESTRUCTIBLE_TOWERS`` take part, and only
        after ``TOWER_READS_BEFORE_DESTRUCTIBLE`` successful reads, so neither
        a king's ordinary silence nor one stray reading can destroy anything.

        A destroyed tower keeps being watched rather than being written off,
        because absence is weak evidence and rubble shows no number: readings
        that arrive afterwards retract the destruction. See
        :meth:`_retract_tower_destroyed`.
        """
        if key not in self.tower_hp:
            raise KeyError(f"Unknown tower key {key!r}; expected one of {TOWER_KEYS}")
        if value is not None and value <= 0:
            # Not a real reading - no tower ever displays 0 - so treat it as
            # an unreadable frame rather than as a destroyed tower.
            value = None
        if key in self._destroyed_towers:
            # Keep watching it. Rubble never shows a number, so readings here
            # disprove the destruction rather than update it.
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
                    self.tower_hp[key] = 0
                    self._register_tower_destroyed(key)
            return  # otherwise keep the last known HP
        self._missing_reads[key] = 0
        self._read_counts[key] += 1
        # HP only ever falls, so the largest reading of the match is this
        # tower's maximum. Learned rather than assumed because the enemy's
        # tower level is unknowable before the match and varies per opponent.
        if value > self.tower_max_hp[key]:
            self.tower_max_hp[key] = value
        self.tower_hp[key] = value

    def update_tower_hp(self, readings: dict[str, Optional[int]]) -> None:
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
        v = self.tower_hp["enemy_left"]
        return v is None or v > 0

    def is_enemy_right_alive(self) -> bool:
        v = self.tower_hp["enemy_right"]
        return v is None or v > 0

    def is_enemy_king_active(self) -> bool:
        """Whether the enemy king tower is firing.

        Two independent signals, matching the game's own rules. The king
        activates when it takes damage - and it only draws its HP number
        once damaged, so having ever read that number means it is active.
        It also activates when either enemy princess tower falls, which no
        HP reading would reveal.

        Comparing HP against max would not work: the king's max is learned
        from its readings, and its first reading is already post-damage, so
        the two are equal exactly when it has just activated.
        """
        return (
            self._read_counts["enemy_king"] >= TOWER_READS_BEFORE_DESTRUCTIBLE
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
