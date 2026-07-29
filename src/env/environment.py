"""Top-level orchestration: ``ClashEnv``.

This is the single integration point ML code talks to::

    env = ClashEnv("BlueStacks App Player 1", model_id="troop-counter/7")
    obs = env.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
    env.close()

Each ``step`` does a full perception cycle (capture -> infer -> update
board, state, lifecycle, towers), executes the action, and returns a
fresh observation plus a reward.

The default reward is ``delta_enemy_hp - delta_friendly_hp`` per step
plus ``+10/-10`` on win/loss. Pass a custom ``reward_fn`` if you want
shaped rewards.

Optional ``record_path`` writes a JSONL recording suitable for
imitation learning. Observations and actions are two *separate*
timestamped streams (``{"type": "obs"}`` / ``{"type": "act"}`` lines)
rather than one line per step, because a human can place several cards
inside a single ~0.3s perception cycle and a one-action-per-step record
silently mis-pairs them. Readers pair offline: every action attaches to
the nearest observation preceding its timestamp. See
``open_record_jsonl`` for the schema header and
``src/imitation/dataset.py`` for the pairing.

A recording run also emits one ``{"type": "diag"}`` line per perception
cycle holding the raw lifecycle decision inputs (elixir magenta
fraction, template scores, banner pixel and colour distances, and which
code path produced the verdict). It is a separate line type, not extra
observation fields, so the observation schema hash stays put; readers
ignore unknown types, so it needs no ``record_format`` bump.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional

import numpy as np

from .actions import (
    Action,
    ActionExecutor,
    ActionResult,
    action_space_size,
    action_to_index,
    index_to_action,
)
from ..vision.capture import ScreenCapture
from ..game.board import GameBoard
from ..game.state import GameState
from ..vision.lifecycle import (
    LifecycleSignals,
    MatchLifecycle,
    STATE_IN_MATCH,
    STATE_POSTMATCH,
    TEMPLATE_MATCH_THRESHOLD,
)
from .observation import ObservationBuilder, schema_descriptor, schema_hash
from ..vision.ocr import MatchTimerReader, TowerHealthReader

# Roboflow inference is heavy; import lazily inside ``_load_model`` so
# tests / static checks can import this module without the SDK.


# prev_tower_hp is a copy of state.tower_hp taken before the step.
RewardFn = Callable[
    [dict[str, Optional[int]], GameState, Optional[str], "ActionResult"], float
]


def default_reward(
    prev_tower_hp: dict[str, Optional[int]],
    state: GameState,
    result: Optional[str],
    action_result: ActionResult,
) -> float:
    """0.001 per HP knocked off enemy towers, -0.001 per friendly HP lost,
    +/-10 on win/loss, -0.05 for a failed (non-no-op) action."""

    def total(side: str, tower_hp: dict[str, Optional[int]]) -> float:
        return sum(
            float(v)
            for k, v in tower_hp.items()
            if k.startswith(side) and v is not None
        )

    delta_enemy = total("enemy", prev_tower_hp) - total("enemy", state.tower_hp)
    delta_friendly = total("friendly", prev_tower_hp) - total(
        "friendly", state.tower_hp
    )

    reward = 0.001 * (delta_enemy - delta_friendly)

    if result == "win":
        reward += 10.0
    elif result == "loss":
        reward -= 10.0

    if not action_result.success:
        reward -= 0.05

    return reward


# On-disk record framing version. 1 = one flat line per step with the
# action embedded (pre-2026-07; mis-pairs multi-placement cycles).
# 2 = separate timestamped "obs" and "act" lines, paired offline.
# Bumped independently of the observation schema hash: the observation
# *layout* is unchanged, only how records are framed, so trained
# checkpoints stay valid while old recordings are refused.
RECORD_FORMAT = 2

# How long a match may run without the on-screen timer ever being read
# before we give up. The clock starts when the elixir bar appears, a few
# seconds before the real match does, so the first frames legitimately
# fail: the timer sits frozen at 3:00 (unusable as an anchor) or is hidden
# behind the countdown overlay. Failing for this long means
# MATCH_TIMER_REGION is miscalibrated, and running on an unanchored clock
# is exactly the ~5s-fast simulation this anchoring exists to remove.
CLOCK_ANCHOR_GRACE_SEC = 20.0


def check_record_meta(meta: dict, path: str) -> None:
    """Raise unless ``meta`` declares the current record framing version.

    Format-1 files have no ``record_format`` key at all; they store the
    action inline on each step line, so a format-2 reader would see zero
    actions rather than an error. Refuse them explicitly.
    """
    found = meta.get("record_format")
    if found != RECORD_FORMAT:
        raise ValueError(
            f"{path} uses record format {found!r} but this code writes and "
            f"reads format {RECORD_FORMAT}. Format 1 packed the action into "
            f"each step line and mis-paired cycles with multiple "
            f"placements; format 2 writes separate timestamped 'obs' and "
            f"'act' lines. Re-record; format-1 files cannot be paired "
            f"reliably after the fact (their actions carry no timestamp)."
        )


def write_record(f, record: dict) -> None:
    """Append one JSON line and flush (recordings must survive a crash)."""
    f.write(json.dumps(record) + "\n")
    f.flush()


def open_record_jsonl(path: str):
    """Open a JSONL recording for append, enforcing schema consistency.

    A new (or empty) file gets a ``{"type": "meta", ...}`` header line
    carrying the record framing version, the observation schema hash and
    the full descriptor (troop class list + field shapes), so future
    readers can detect layout changes and migrate old data. Appending to
    a file recorded under a different observation schema or a different
    record format raises instead of silently mixing layouts.
    """
    os.makedirs(
        os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True
    )
    current = schema_hash()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        if first.get("type") != "meta":
            raise ValueError(
                f"{path} has no schema header (recorded before schema "
                f"versioning). Record to a new file instead of appending."
            )
        check_record_meta(first, path)
        if first.get("schema_hash") != current:
            raise ValueError(
                f"{path} was recorded under observation schema "
                f"{first['schema_hash']} but the current schema is "
                f"{current}. Record to a new file; the old file's meta "
                f"header retains its troop class list for migration."
            )
        return open(path, "a", encoding="utf-8")

    f = open(path, "a", encoding="utf-8")
    write_record(
        f,
        {
            "type": "meta",
            "record_format": RECORD_FORMAT,
            "schema_hash": current,
            "schema": schema_descriptor(),
        },
    )
    return f


def obs_record(
    *,
    t: float,
    step: int,
    obs_flat: list[float],
    reward: float,
    signals: LifecycleSignals,
    state: GameState,
    source: str,
) -> dict:
    """One observation line. ``t`` is the frame *capture* time — the
    moment the recorded screen state actually existed, which is what
    action pairing keys off."""
    return {
        "type": "obs",
        "t": t,
        "step": step,
        "obs_flat": obs_flat,
        "reward": reward,
        "lifecycle_state": signals.state,
        "lifecycle_result": signals.result,
        "match_time": state.get_current_match_time(),
        "elixir": state.get_current_elixir(),
        "match_result": state.match_result,
        "source": source,
    }


def diag_record(*, t: float, step: int, signals: LifecycleSignals) -> dict:
    """One lifecycle-diagnostics line: the raw inputs behind this cycle's
    lifecycle verdict, stamped with the same frame-capture ``t`` as the
    observation line for the same cycle.

    A third line type rather than extra observation fields, and no
    ``record_format`` bump: the observation layout is hashed into every
    recording's meta header and into trained checkpoints, so growing it
    would invalidate both, whereas an *additional* optional line type
    leaves every existing obs/act line byte-identical and readers that
    don't know it skip it (see ``src/imitation/dataset.py``).

    ``template_hits`` is ``{}`` and the banner fields are ``None`` when
    the elixir gate short-circuited before those signals were computed —
    absence here is itself the diagnosis.
    """
    banner = signals.banner
    return {
        "type": "diag",
        "t": t,
        "step": step,
        "path": signals.path,
        "lifecycle_state": signals.state,
        "lifecycle_result": signals.result,
        "elixir_magenta_frac": signals.elixir_magenta_frac,
        "elixir_bar_visible": signals.elixir_bar_visible,
        "template_hits": signals.template_hits,
        "banner_bgr": list(banner.bgr) if banner is not None else None,
        "banner_dist_victory": banner.dist_victory if banner is not None else None,
        "banner_dist_defeat": banner.dist_defeat if banner is not None else None,
    }


def act_record(
    *,
    t: float,
    action: Action,
    success: bool,
    reason: str,
    source: str,
) -> dict:
    """One action line. ``t`` is when the action was issued (bot) or when
    the drag was released (human). No-ops are never written — an
    observation with no action attached *is* the no-op sample."""
    if action.is_no_op:
        raise ValueError("no-op actions are not recorded; skip the write")
    return {
        "type": "act",
        "t": t,
        "action_index": action_to_index(action),
        "hand_index": action.hand_index,
        "tile_x": action.tile_x,
        "tile_y": action.tile_y,
        "success": success,
        "reason": reason,
        "source": source,
    }


class ClashEnv:
    def __init__(
        self,
        window_title: str,
        model_id: str = "troop-counter/7",
        api_key: Optional[str] = None,
        record_path: Optional[str] = None,
        reward_fn: Optional[RewardFn] = None,
        step_period_sec: float = 0.25,
        max_match_duration_sec: float = 320.0,
    ):
        self.window_title = window_title
        self.model_id = model_id
        self.api_key = api_key
        self.record_path = record_path
        self.reward_fn = reward_fn or default_reward
        self.step_period_sec = step_period_sec
        self.max_match_duration_sec = max_match_duration_sec

        self.capture: Optional[ScreenCapture] = None
        self.board: Optional[GameBoard] = None
        self.state = GameState()
        self.lifecycle = MatchLifecycle()
        self.tower_reader = TowerHealthReader()
        self.timer_reader = MatchTimerReader()
        self.observer = ObservationBuilder()
        self.executor = ActionExecutor()

        self._model = None
        self._supervision = None
        self._frame: Optional[np.ndarray] = None
        # Wall-clock time the last frame was grabbed. Recordings stamp
        # observations with this, not with the (much later) write time.
        self.frame_time: float = 0.0
        self._last_step_time: float = 0.0
        self._record_file = None
        self._step_count: int = 0

    # ----- model bootstrap -----

    def _load_model(self) -> None:
        if self._model is not None:
            return
        # Suppress unused inference models the same way the legacy script did.
        os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
        os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
        os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
        os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")

        from inference import get_model
        import supervision as sv

        api_key = self.api_key or os.getenv("API_KEY")
        self._model = get_model(model_id=self.model_id, api_key=api_key)
        self._supervision = sv

    # ----- spaces -----

    @property
    def action_space_size(self) -> int:
        return action_space_size()

    # ----- public API -----

    def reset(self, wait_timeout_sec: float = 60.0) -> dict:
        self._load_model()
        if self.capture is None:
            self.capture = ScreenCapture(self.window_title).__enter__()

        self._wait_for_in_match(timeout_sec=wait_timeout_sec)

        monitor = self.capture.monitor
        self.board = GameBoard(monitor["width"], monitor["height"])
        self.state = GameState()
        self.state.start_match()

        if self.record_path:
            self._open_record_file()

        self._step_count = 0

        # Perceive before returning: without this the first observation
        # of every episode is a blank arena with an empty hand, and the
        # first action is chosen from it.
        obs, signals = self.observe()
        self._write_obs_record(obs, reward=0.0, step=0, signals=signals)
        self._write_diag_record(step=0, signals=signals)
        return obs

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        if isinstance(action, (int, np.integer)):
            action_obj = index_to_action(int(action))
        elif isinstance(action, Action):
            action_obj = action
        elif isinstance(action, (tuple, list)) and len(action) == 3:
            action_obj = Action(int(action[0]), int(action[1]), int(action[2]))
        else:
            raise TypeError(f"Unsupported action type: {type(action)!r}")

        self.throttle()

        prev_tower_hp = dict(self.state.tower_hp)

        # Execute action first so vision picks up the new troop next
        # frame. The action is stamped with its *issue* time, which falls
        # between the previous observation's frame time and this step's,
        # so the offline nearest-preceding-observation pairing recovers
        # (s_t, a_t) — the state the policy actually acted on — even
        # though the observation returned below is s_{t+1}.
        assert self.capture is not None and self.board is not None
        action_time = time.time()
        action_result = self.executor.execute(
            action_obj, self.board, self.state, self.capture.monitor
        )
        if not action_obj.is_no_op:
            self._write_act_record(action_time, action_obj, action_result)

        # Capture + perceive after the action lands.
        obs, signals = self.observe()

        done = self.resolve_done(signals)

        reward = float(
            self.reward_fn(prev_tower_hp, self.state, signals.result, action_result)
        )

        info: dict[str, Any] = {
            "action_index": action_to_index(action_obj),
            "action": action_obj,
            "action_result": {
                "success": action_result.success,
                "reason": action_result.reason,
            },
            "lifecycle_state": signals.state,
            "lifecycle_result": signals.result,
            "match_time": self.state.get_current_match_time(),
            "elixir": self.state.get_current_elixir(),
            "match_result": self.state.match_result,
            "step": self._step_count,
        }

        # reset() wrote observation 0, so this step's observation is
        # index _step_count + 1.
        self._write_obs_record(
            obs, reward=reward, step=self._step_count + 1, signals=signals
        )
        self._write_diag_record(step=self._step_count + 1, signals=signals)

        if done:
            self.state.end_match(self.state.match_result)

        self._step_count += 1
        return obs, reward, done, info

    def observe(self) -> tuple[dict, LifecycleSignals]:
        """One perception cycle without acting: grab a frame, refresh
        board / tower HP, detect the lifecycle state, build an
        observation. Used by ``step`` and by passive consumers like the
        human demo recorder."""
        assert self.capture is not None and self.board is not None
        self.frame_time = time.time()
        frame = self.capture.grab()
        self._frame = frame
        self._refresh_perception(frame)
        signals = self.lifecycle.detect_state(frame)
        return self._build_observation(), signals

    def resolve_done(self, signals: LifecycleSignals) -> bool:
        """Adopt a lifecycle-reported result and report whether the
        episode is over. Shared by ``step`` and the human demo recorder
        so both agree on when a match ends."""
        if signals.state == STATE_POSTMATCH:
            if signals.result and self.state.match_result is None:
                self.state.set_match_result(signals.result)
            return True
        if self.state.match_result is not None:
            # Only the lifecycle sets this now, so reaching it means the
            # banner was seen on an earlier frame. It used to be reachable
            # from a single bad tower-HP reading, which ended matches
            # mid-play while this very check said the state was IN_MATCH.
            return True
        # Uncapped elapsed, not get_current_match_time(): that saturates at
        # MATCH_MAX_DURATION (300s), so comparing it against a longer
        # timeout could never fire and a match with no postmatch screen
        # would run forever.
        return self.state.get_elapsed_seconds() >= self.max_match_duration_sec

    def close(self) -> None:
        if self._record_file is not None:
            self._record_file.close()
            self._record_file = None
        self.tower_reader.close()
        self.timer_reader.close()
        if self.capture is not None:
            self.capture.__exit__(None, None, None)
            self.capture = None

    def throttle(self) -> None:
        """Sleep just long enough that consecutive calls are
        ``step_period_sec`` apart. Subtractive, not additive: perception
        already eats most of the period. Public so the human demo
        recorder paces itself exactly like the bot does."""
        now = time.time()
        if self._last_step_time:
            wait = self.step_period_sec - (now - self._last_step_time)
            if wait > 0:
                time.sleep(wait)
        self._last_step_time = time.time()

    # ----- internals -----

    def _wait_for_in_match(self, timeout_sec: float) -> None:
        """Drive the UI into a match: dismiss postmatch screens, press
        Battle when the menu button is visible, then wait for the match
        to start. Battle is only clicked on a confirmed template hit so
        we never click blindly during loading or matchmaking."""
        assert self.capture is not None
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            frame = self.capture.grab()
            signals = self.lifecycle.detect_state(frame)
            if signals.state == STATE_IN_MATCH:
                return
            monitor = self.capture.monitor
            if signals.state == STATE_POSTMATCH:
                self.lifecycle.click_ok(monitor, frame)
                time.sleep(2.0)
            elif (
                signals.template_hits.get("battle_button", 0.0)
                >= TEMPLATE_MATCH_THRESHOLD
            ):
                self.lifecycle.click_battle(monitor, frame)
                time.sleep(2.0)
            else:
                time.sleep(0.5)
        raise TimeoutError(
            f"Did not detect match start within {timeout_sec:.0f}s"
        )

    def _refresh_perception(self, frame: np.ndarray) -> None:
        assert self.board is not None and self._supervision is not None
        results = self._model.infer(frame)[0]
        detections = self._supervision.Detections.from_inference(results)
        self.board.clear_arena()
        self.board.clear_hand()
        self.board.process_detections(detections)
        readings = self.tower_reader.read(frame)
        self.state.update_tower_hp(readings)
        if not self.state.clock_anchored:
            self._anchor_clock(frame)

    def _anchor_clock(self, frame: np.ndarray) -> None:
        """Pin the match clock to the on-screen timer, once per match.

        Only runs while the clock is unanchored, so the extra OCR pass
        costs a handful of frames at the start of a match rather than one
        per cycle. Individual unreadable frames are expected and ignored;
        never getting a usable reading is a calibration failure and raises.
        """
        remaining = self.timer_reader.read(frame)
        if remaining is not None and self.state.anchor_match_clock(remaining):
            return
        if self.state.get_elapsed_seconds() > CLOCK_ANCHOR_GRACE_SEC:
            raise RuntimeError(
                f"Could not anchor the match clock to the on-screen timer "
                f"within {CLOCK_ANCHOR_GRACE_SEC:.0f}s (last read: "
                f"{remaining!r}). Match time would be pure simulation, "
                f"running ~5s ahead of the game. Calibrate "
                f"MATCH_TIMER_REGION in src/vision/ocr.py to the 'm:ss' "
                f"countdown in the captured frame."
            )

    def _build_observation(self) -> dict:
        assert self.board is not None
        return self.observer.build(self.board, self.state)

    # ----- record file -----

    def _open_record_file(self) -> None:
        if self._record_file is not None:
            return
        self._record_file = open_record_jsonl(self.record_path)

    def _write_obs_record(
        self, obs: dict, reward: float, step: int, signals: LifecycleSignals
    ) -> None:
        if self._record_file is None:
            return
        write_record(
            self._record_file,
            obs_record(
                t=self.frame_time,
                step=step,
                obs_flat=self.observer.flatten(obs).tolist(),
                reward=reward,
                signals=signals,
                state=self.state,
                source="bot",
            ),
        )

    def _write_diag_record(self, step: int, signals: LifecycleSignals) -> None:
        if self._record_file is None:
            return
        write_record(
            self._record_file, diag_record(t=self.frame_time, step=step, signals=signals)
        )

    def _write_act_record(
        self, t: float, action: Action, result: ActionResult
    ) -> None:
        if self._record_file is None:
            return
        write_record(
            self._record_file,
            act_record(
                t=t,
                action=action,
                success=result.success,
                reason=result.reason,
                source="bot",
            ),
        )
