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

Optional ``record_path`` writes a JSONL line per step suitable for
imitation learning. Each line contains the flat observation, action
index, raw reward, and lifecycle metadata.
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
from ..vision.ocr import TowerHealthReader

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


def open_record_jsonl(path: str):
    """Open a JSONL recording for append, enforcing schema consistency.

    A new (or empty) file gets a ``{"type": "meta", ...}`` header line
    carrying the observation schema hash and full descriptor (troop
    class list + field shapes), so future readers can detect layout
    changes and migrate old data. Appending to a file recorded under a
    different schema raises instead of silently mixing layouts.
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
        if first.get("schema_hash") != current:
            raise ValueError(
                f"{path} was recorded under observation schema "
                f"{first['schema_hash']} but the current schema is "
                f"{current}. Record to a new file; the old file's meta "
                f"header retains its troop class list for migration."
            )
        return open(path, "a", encoding="utf-8")

    f = open(path, "a", encoding="utf-8")
    f.write(
        json.dumps(
            {
                "type": "meta",
                "schema_hash": current,
                "schema": schema_descriptor(),
            }
        )
        + "\n"
    )
    f.flush()
    return f


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
        self.observer = ObservationBuilder()
        self.executor = ActionExecutor()

        self._model = None
        self._supervision = None
        self._frame: Optional[np.ndarray] = None
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
        return self._build_observation()

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        if isinstance(action, (int, np.integer)):
            action_obj = index_to_action(int(action))
        elif isinstance(action, Action):
            action_obj = action
        elif isinstance(action, (tuple, list)) and len(action) == 3:
            action_obj = Action(int(action[0]), int(action[1]), int(action[2]))
        else:
            raise TypeError(f"Unsupported action type: {type(action)!r}")

        self._throttle()

        prev_tower_hp = dict(self.state.tower_hp)

        # Execute action first so vision picks up the new troop next frame.
        assert self.capture is not None and self.board is not None
        action_result = self.executor.execute(
            action_obj, self.board, self.state, self.capture.monitor
        )

        # Capture + perceive after the action lands.
        obs, signals = self.observe()

        # Determine done.
        done = False
        if signals.state == STATE_POSTMATCH:
            done = True
            if signals.result and self.state.match_result is None:
                self.state.set_match_result(signals.result)
        elif self.state.match_result is not None:
            done = True
        elif self.state.get_current_match_time() >= self.max_match_duration_sec:
            done = True

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

        if self.record_path:
            self._write_record(obs, action_obj, reward, info)

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
        frame = self.capture.grab()
        self._frame = frame
        self._refresh_perception(frame)
        signals = self.lifecycle.detect_state(frame)
        return self._build_observation(), signals

    def close(self) -> None:
        if self._record_file is not None:
            self._record_file.close()
            self._record_file = None
        if self.capture is not None:
            self.capture.__exit__(None, None, None)
            self.capture = None

    # ----- internals -----

    def _throttle(self) -> None:
        now = time.time()
        if self._last_step_time:
            wait = self.step_period_sec - (now - self._last_step_time)
            if wait > 0:
                time.sleep(wait)
        self._last_step_time = time.time()

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

    def _build_observation(self) -> dict:
        assert self.board is not None
        return self.observer.build(self.board, self.state)

    # ----- record file -----

    def _open_record_file(self) -> None:
        if self._record_file is not None:
            return
        self._record_file = open_record_jsonl(self.record_path)

    def _write_record(self, obs: dict, action: Action, reward: float, info: dict) -> None:
        if self._record_file is None:
            return
        flat = self.observer.flatten(obs)
        record = {
            "t": time.time(),
            "step": info["step"],
            "obs_flat": flat.tolist(),
            "action_index": info["action_index"],
            "hand_index": action.hand_index,
            "tile_x": action.tile_x,
            "tile_y": action.tile_y,
            "reward": reward,
            "lifecycle_state": info["lifecycle_state"],
            "lifecycle_result": info["lifecycle_result"],
            "match_time": info["match_time"],
            "elixir": info["elixir"],
            "match_result": info["match_result"],
            "action_success": info["action_result"]["success"],
            "action_reason": info["action_result"]["reason"],
        }
        self._record_file.write(json.dumps(record) + "\n")
        self._record_file.flush()
