"""Record human demonstrations for imitation learning.

Runs ``ClashEnv``'s perception cycle passively (``env.observe``, never
``env.step``) while a global pynput mouse listener watches the human
play in BlueStacks. Drags are reverse-mapped to ``Action`` objects:

- press near a hand slot centre (``HAND_CARD_POSITIONS``) -> hand_index
- release inside the arena -> (tile_x, tile_y)

Output uses the env's two-stream JSONL format (see
``src/env/environment.py``): timestamped ``{"type": "obs"}`` lines from
the perception loop and timestamped ``{"type": "act"}`` lines from the
mouse listener, plus ``"source": "human"``. The two streams are *not*
paired here. A perception cycle takes ~0.3s, far longer than the gap
between two quick placements, so pairing at record time would drop or
skew actions; ``src/imitation/dataset.py`` pairs them offline by
timestamp instead. Cycles with no action attached become the NO_OP
training samples.

Limitation: only drag-style placement is detected. Tap-the-card then
tap-the-tile placements are ignored, so play by dragging.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional

from pynput import mouse

from ..env.actions import Action, ActionResult, HAND_CARD_POSITIONS
from ..env.environment import (
    ClashEnv,
    act_record,
    default_reward,
    diag_record,
    obs_record,
    open_record_jsonl,
    write_record,
)
from ..game.board import GameBoard
from ..vision.lifecycle import LifecycleSignals

# How close (in monitor fractions) a press must be to a hand-slot centre
# to count as picking up that card. X bound is just under half the
# ~0.19 slot spacing in HAND_CARD_POSITIONS.
HAND_SLOT_TOLERANCE_X = 0.09
HAND_SLOT_TOLERANCE_Y = 0.05

# Releases below this y-fraction are still in the hand area (cards sit
# at ~0.885), not an arena placement.
ARENA_MAX_Y_FRAC = 0.84


@dataclass(frozen=True)
class TimedAction:
    """An ``Action`` plus the wall-clock time it happened.

    Kept here rather than as a field on ``Action`` on purpose: ``Action``
    is the env/policy contract (constructed by policies, compared,
    hashed, converted to and from indices) and a capture timestamp is
    purely a recording concern. Wrapping keeps the contract clean.
    """

    t: float
    action: Action


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
        self._actions: "queue.Queue[TimedAction]" = queue.Queue()
        self._listener = mouse.Listener(on_click=self._on_click)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def clear(self) -> None:
        """Drop actions queued outside an episode (menu clicks etc.)."""
        self.drain()

    def drain(self) -> list[TimedAction]:
        """Every placement queued since the last drain, oldest first.

        Returns all of them — a slow perception cycle routinely spans
        several placements and dropping the extras (or deferring them to
        a later cycle) is exactly the mis-pairing this format exists to
        avoid.
        """
        out: list[TimedAction] = []
        while True:
            try:
                out.append(self._actions.get_nowait())
            except queue.Empty:
                return out

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

        # Stamp the release immediately: this is the moment the placement
        # completed, and it must come from the same clock as the "t" on
        # observation records for offline pairing to work.
        released_at = time.time()

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
        self._actions.put(
            TimedAction(
                t=released_at,
                action=Action(hand_index=slot, tile_x=tile[0], tile_y=tile[1]),
            )
        )


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
            env.reset()
            watcher.clear()
            _record_episode(env, watcher, out)
    finally:
        watcher.stop()
        out.close()
        env.close()


def _write_obs(
    out,
    env: ClashEnv,
    obs: dict,
    signals: LifecycleSignals,
    step: int,
    reward: float,
) -> None:
    write_record(
        out,
        obs_record(
            t=env.frame_time,
            step=step,
            obs_flat=env.observer.flatten(obs).tolist(),
            reward=reward,
            signals=signals,
            state=env.state,
            source="human",
        ),
    )
    # Same diagnostics line ``ClashEnv.step`` writes. Human demos went
    # without it, which is why the first session's three mislabelled
    # outcomes had to be diagnosed from a screenshot instead of from the
    # template scores on the frames that actually decided them.
    write_record(out, diag_record(t=env.frame_time, step=step, signals=signals))


def _record_episode(env: ClashEnv, watcher: MouseWatcher, out) -> None:
    """Perceive in a loop, writing an obs line per cycle and an act line
    per detected drag, until the match ends.

    ``env.reset`` already perceived once but its record went to the env's
    own file (unused here), so the episode's first observation is taken
    fresh below.
    """
    step = 0
    placements = 0

    obs, signals = env.observe()
    _write_obs(out, env, obs, signals, step=step, reward=0.0)
    done = env.resolve_done(signals)

    while not done:
        env.throttle()

        for timed in watcher.drain():
            placements += 1
            card = env.board.cards_in_hand[timed.action.hand_index]
            if card is not None:
                env.state.spend_elixir(card.cost)
            else:
                print(
                    f"WARNING: placement from slot {timed.action.hand_index} "
                    f"but vision sees no card there; elixir not deducted"
                )
            write_record(
                out,
                act_record(
                    t=timed.t,
                    action=timed.action,
                    success=True,
                    reason="human",
                    source="human",
                ),
            )

        prev_tower_hp = dict(env.state.tower_hp)
        obs, signals = env.observe()
        done = env.resolve_done(signals)

        # Terminal term on the closing step only, from the result the env
        # adopted rather than this frame's signals — the outcome is often
        # read a frame or two into the postmatch settle window.
        reward = default_reward(
            prev_tower_hp,
            env.state,
            env.state.match_result if done else None,
            ActionResult(success=True, reason="human"),
        )

        step += 1
        _write_obs(out, env, obs, signals, step=step, reward=reward)

    env.state.end_match(env.state.match_result)
    print(
        f"Episode recorded: observations={step + 1} placements={placements} "
        f"result={env.state.match_result}"
    )
