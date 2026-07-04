"""Record human demonstrations for imitation learning.

Runs ``ClashEnv``'s perception cycle passively (``env.observe``, never
``env.step``) while a global pynput mouse listener watches the human
play in BlueStacks. Drags are reverse-mapped to ``Action`` objects:

- press near a hand slot centre (``HAND_CARD_POSITIONS``) -> hand_index
- release inside the arena -> (tile_x, tile_y)

Steps with no drag are recorded as NO_OP. Each record uses the same
JSONL schema as ``ClashEnv``'s ``record_path`` plus ``"source":
"human"``, so dataset code can consume bot and human logs alike. Unlike
``ClashEnv`` recording, the observation in each record is the one *seen
before* the action was taken — the pairing behaviour cloning needs.

Limitation: only drag-style placement is detected. Tap-the-card then
tap-the-tile placements are ignored, so play by dragging.
"""

from __future__ import annotations

import json
import queue
import time
from typing import Callable, Optional

from pynput import mouse

from ..env.actions import (
    Action,
    ActionResult,
    action_to_index,
    HAND_CARD_POSITIONS,
)
from ..env.environment import ClashEnv, default_reward, open_record_jsonl
from ..game.board import GameBoard
from ..vision.lifecycle import STATE_POSTMATCH

# How close (in monitor fractions) a press must be to a hand-slot centre
# to count as picking up that card. X bound is just under half the
# ~0.19 slot spacing in HAND_CARD_POSITIONS.
HAND_SLOT_TOLERANCE_X = 0.09
HAND_SLOT_TOLERANCE_Y = 0.05

# Releases below this y-fraction are still in the hand area (cards sit
# at ~0.885), not an arena placement.
ARENA_MAX_Y_FRAC = 0.84


class MouseWatcher:
    """Global mouse listener that turns hand->arena drags into Actions.

    ``get_monitor`` / ``get_board`` are callables because the capture
    monitor tracks the live window position and the board is recreated
    each episode. Events arriving while either is unavailable are
    dropped.
    """

    def __init__(
        self,
        get_monitor: Callable[[], Optional[dict]],
        get_board: Callable[[], Optional[GameBoard]],
    ):
        self._get_monitor = get_monitor
        self._get_board = get_board
        self._press_frac: Optional[tuple[float, float]] = None
        self._actions: "queue.Queue[Action]" = queue.Queue()
        self._listener = mouse.Listener(on_click=self._on_click)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def clear(self) -> None:
        """Drop actions queued outside an episode (menu clicks etc.)."""
        while True:
            try:
                self._actions.get_nowait()
            except queue.Empty:
                return

    def pop(self) -> Action:
        """The oldest placement since the last pop, or NO_OP."""
        try:
            return self._actions.get_nowait()
        except queue.Empty:
            return Action.no_op()

    # ----- listener internals (run on the pynput thread) -----

    def _to_frac(self, x: float, y: float) -> Optional[tuple[float, float]]:
        monitor = self._get_monitor()
        if monitor is None:
            return None
        return (
            (x - monitor["left"]) / monitor["width"],
            (y - monitor["top"]) / monitor["height"],
        )

    @staticmethod
    def _hand_slot(frac: tuple[float, float]) -> Optional[int]:
        for i, (fx, fy) in enumerate(HAND_CARD_POSITIONS):
            if (
                abs(frac[0] - fx) <= HAND_SLOT_TOLERANCE_X
                and abs(frac[1] - fy) <= HAND_SLOT_TOLERANCE_Y
            ):
                return i
        return None

    def _on_click(self, x: float, y: float, button, pressed: bool) -> None:
        if button != mouse.Button.left:
            return
        if pressed:
            self._press_frac = self._to_frac(x, y)
            return

        press, self._press_frac = self._press_frac, None
        if press is None:
            return
        slot = self._hand_slot(press)
        if slot is None:
            return
        release = self._to_frac(x, y)
        if release is None or release[1] > ARENA_MAX_Y_FRAC:
            return  # released back in the hand / UI area, not a placement

        monitor = self._get_monitor()
        board = self._get_board()
        if monitor is None or board is None:
            return
        tile = board.convert_image_cord_to_tile(
            release[0] * monitor["width"], release[1] * monitor["height"]
        )
        if tile is None:
            return
        self._actions.put(Action(hand_index=slot, tile_x=tile[0], tile_y=tile[1]))


def record_demos(
    window_title: str,
    model_id: str,
    record_path: str,
    episodes: int,
) -> None:
    """Record ``episodes`` matches of human play to ``record_path``.

    The env auto-drives menu/postmatch screens (Battle / OK clicks); the
    human only plays the matches.
    """
    env = ClashEnv(window_title=window_title, model_id=model_id)
    watcher = MouseWatcher(
        get_monitor=lambda: env.capture.monitor if env.capture else None,
        get_board=lambda: env.board,
    )

    out = open_record_jsonl(record_path)
    watcher.start()

    try:
        for episode in range(episodes):
            print(
                f"\n=== Recording episode {episode + 1}/{episodes} — "
                f"play in BlueStacks (drag cards from hand to arena) ==="
            )
            obs = env.reset()
            watcher.clear()
            _record_episode(env, watcher, out)
    finally:
        watcher.stop()
        out.close()
        env.close()


def _record_episode(env: ClashEnv, watcher: MouseWatcher, out) -> None:
    step = 0
    placements = 0
    done = False

    while not done:
        time.sleep(env.step_period_sec)
        action = watcher.pop()

        if not action.is_no_op:
            placements += 1
            card = env.board.cards_in_hand[action.hand_index]
            if card is not None:
                env.state.spend_elixir(card.cost)
            else:
                print(
                    f"WARNING: placement from slot {action.hand_index} but "
                    f"vision sees no card there; elixir not deducted"
                )

        prev_tower_hp = dict(env.state.tower_hp)
        next_obs, signals = env.observe()

        if signals.state == STATE_POSTMATCH:
            done = True
            if signals.result and env.state.match_result is None:
                env.state.set_match_result(signals.result)
        elif env.state.match_result is not None:
            done = True
        elif env.state.get_current_match_time() >= env.max_match_duration_sec:
            done = True

        reward = default_reward(
            prev_tower_hp,
            env.state,
            signals.result,
            ActionResult(success=True, reason="human"),
        )

        record = {
            "t": time.time(),
            "step": step,
            "obs_flat": env.observer.flatten(obs).tolist(),
            "action_index": action_to_index(action),
            "hand_index": action.hand_index,
            "tile_x": action.tile_x,
            "tile_y": action.tile_y,
            "reward": reward,
            "lifecycle_state": signals.state,
            "lifecycle_result": signals.result,
            "match_time": env.state.get_current_match_time(),
            "elixir": env.state.get_current_elixir(),
            "match_result": env.state.match_result,
            "action_success": True,
            "action_reason": "human",
            "source": "human",
        }
        out.write(json.dumps(record) + "\n")
        out.flush()

        obs = next_obs
        step += 1

    env.state.end_match(env.state.match_result)
    print(
        f"Episode recorded: steps={step} placements={placements} "
        f"result={env.state.match_result}"
    )
