"""Visual debugger for the simulator.

Draws two panels side by side, plus an optional third:

- TRUTH: where entities actually are, at full float precision.
- OBSERVED: the tile grid the policy is handed, after detection dropout,
  position jitter, phantom units and stale tower bars.
- REWARD: the running return split by SOURCE, when a ``RewardTrace`` is
  supplied.

The first pair is the point. Sim fidelity and perception noise are the two
things that decide whether a policy transfers, and both are invisible in a
reward curve. Seeing a unit vanish from the right panel while it is plainly
there on the left is the fastest way to understand what the policy is working
with.

The reward panel is the same ``reward_parts`` audit that found the sparsity
problem, watched live instead of aggregated: a single scalar cannot show that
91% of steps paid exactly nothing.

Rendering only - it never advances the sim, so it cannot perturb a run.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..game.board import ARENA_COLS, ARENA_ROWS
from ..game.classes import normalize_name
from ..game.state import TOWER_KEYS
from . import arena
from .engine import Simulation
from .units import SPELL_NAMES

TILE = 44                      # pixels per grid tile
HUD_H = 96
PANEL_W = ARENA_COLS * TILE
PANEL_H = ARENA_ROWS * TILE
GAP = 12
REWARD_W = 300                 # width of the optional reward panel

#: How many recent placements stay marked on the TRUTH panel. The open
#: question about this policy is WHERE it puts cards, so the last few
#: choices are worth more than the current one alone.
PLACEMENT_TRAIL = 6

COL_BG = (34, 30, 28)
COL_GRASS = (48, 82, 46)
COL_GRASS_ALT = (52, 88, 50)
COL_RIVER = (120, 84, 40)
COL_BRIDGE = (60, 92, 128)
COL_GRID = (70, 70, 70)
COL_FRIENDLY = (235, 170, 60)   # BGR: cyan-ish blue, matches the overlay
COL_ENEMY = (70, 70, 235)
COL_TEXT = (235, 235, 235)
COL_DIM = (150, 150, 150)
COL_HP_GOOD = (90, 210, 90)
COL_HP_LOW = (60, 60, 230)
COL_GHOST = (0, 235, 235)
COL_GAIN = (110, 220, 110)
COL_LOSS = (90, 90, 240)
COL_PLACE = (255, 255, 255)


class RewardTrace:
    """Per-episode reward bookkeeping for the overlay.

    Display state, owned by the viewer - the env never sees it, so the
    overlay cannot perturb a run. Holds the running return, the cumulative
    split by source, a short history for the sparkline, and the recent
    placements.

    Part ORDER comes from the env's own ``reward_parts`` dict, which returns
    every key every step. Re-listing the keys here would be a second copy of
    that contract, free to drift.
    """

    HISTORY = 160

    def __init__(self) -> None:
        self.opponent = ""
        self.reset()

    def reset(self, opponent: str = "") -> None:
        self.opponent = opponent or self.opponent
        self.total = 0.0
        self.last = 0.0
        self.step = 0
        self.parts: dict[str, float] = {}
        self.live: dict[str, float] = {}
        self.history: deque[float] = deque(maxlen=self.HISTORY)
        self.placements: list[tuple[int, int, int, bool]] = []
        self.silent_steps = 0

    def update(self, reward: float, info: dict, action=None) -> None:
        self.total += reward
        self.last = float(reward)
        self.step = int(info.get("step", self.step + 1))
        self.history.append(float(reward))
        if abs(reward) < 1e-9:
            self.silent_steps += 1

        parts = info.get("reward_parts", {})
        for key, value in parts.items():
            self.parts[key] = self.parts.get(key, 0.0) + value
        # Only the sources that actually paid THIS step, for the flash.
        self.live = {k: v for k, v in parts.items() if abs(v) > 1e-9}

        if action is not None and not action.is_no_op:
            self.placements.append((
                int(action.tile_x), int(action.tile_y), self.step,
                bool(info.get("action_ok", True)),
            ))
            del self.placements[:-PLACEMENT_TRAIL]

    @property
    def silent_fraction(self) -> float:
        return self.silent_steps / self.step if self.step else 0.0


def _tile_px(x: float, y: float) -> tuple[int, int]:
    return int(x * TILE), int(y * TILE)


def _draw_field(panel: np.ndarray) -> None:
    for ty in range(ARENA_ROWS):
        for tx in range(ARENA_COLS):
            c = COL_GRASS if (tx + ty) % 2 == 0 else COL_GRASS_ALT
            cv2.rectangle(
                panel, (tx * TILE, ty * TILE),
                ((tx + 1) * TILE, (ty + 1) * TILE), c, -1,
            )

    y0 = int((arena.RIVER_Y - arena.RIVER_HALF_WIDTH) * TILE)
    y1 = int((arena.RIVER_Y + arena.RIVER_HALF_WIDTH) * TILE)
    cv2.rectangle(panel, (0, y0), (PANEL_W, y1), COL_RIVER, -1)
    for bx in arena.BRIDGE_X:
        x0 = int((bx - arena.BRIDGE_HALF_WIDTH) * TILE)
        x1 = int((bx + arena.BRIDGE_HALF_WIDTH) * TILE)
        cv2.rectangle(panel, (x0, y0), (x1, y1), COL_BRIDGE, -1)

    for tx in range(ARENA_COLS + 1):
        cv2.line(panel, (tx * TILE, 0), (tx * TILE, PANEL_H), COL_GRID, 1)
    for ty in range(ARENA_ROWS + 1):
        cv2.line(panel, (0, ty * TILE), (PANEL_W, ty * TILE), COL_GRID, 1)


def _draw_hp_bar(
    panel: np.ndarray, cx: int, cy: int, frac: float, width: int
) -> None:
    h = 5
    x0, y0 = cx - width // 2, cy - 2
    cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + h), (25, 25, 25), -1)
    if frac > 0:
        col = COL_HP_GOOD if frac > 0.35 else COL_HP_LOW
        cv2.rectangle(
            panel, (x0, y0), (x0 + int(width * frac), y0 + h), col, -1
        )


def _draw_towers(panel: np.ndarray, sim: Simulation, hp: dict[str, float]) -> None:
    for spec in arena.TOWERS:
        frac = hp.get(spec.key, 0.0)
        px, py = _tile_px(spec.x, spec.y)
        # Radius is a level-dependent stat now, so take it off the live
        # entity rather than the position-only spec.
        entity = next(
            (e for e in sim.entities.values() if e.tower_key == spec.key), None
        )
        r = int((entity.radius if entity else 0.5) * TILE)
        col = COL_FRIENDLY if spec.friendly else COL_ENEMY
        if frac <= 0.0:
            cv2.rectangle(panel, (px - r, py - r), (px + r, py + r), (60, 60, 60), -1)
            cv2.line(panel, (px - r, py - r), (px + r, py + r), (30, 30, 30), 2)
            cv2.line(panel, (px - r, py + r), (px + r, py - r), (30, 30, 30), 2)
            continue
        cv2.rectangle(panel, (px - r, py - r), (px + r, py + r), col, 2)
        _draw_hp_bar(panel, px, py + r + 8, frac, r * 2)
        if spec.is_king:
            if entity is not None and not entity.active:
                cv2.putText(panel, "zzz", (px - 14, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DIM, 1)


def _draw_placements(panel: np.ndarray, trace: RewardTrace) -> None:
    """Mark where the policy has recently put cards.

    Fades with age so the newest is obvious. A REJECTED placement draws a
    cross instead of a ring - that is the ``invalid_action`` penalty made
    visible at the tile it was paid for.
    """
    for tx, ty, step, ok in trace.placements:
        age = max(0, trace.step - step)
        fade = max(0.25, 1.0 - age / (PLACEMENT_TRAIL * 8))
        col = tuple(int(c * fade) for c in (COL_PLACE if ok else COL_LOSS))
        px, py = _tile_px(tx + 0.5, ty + 0.5)
        if ok:
            cv2.circle(panel, (px, py), int(TILE * 0.42), col, 2)
        else:
            cv2.drawMarker(panel, (px, py), col, cv2.MARKER_TILTED_CROSS, 20, 2)


def _draw_truth(sim: Simulation, trace: RewardTrace | None = None) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    _draw_field(panel)
    _draw_towers(panel, sim, sim.tower_hp_fractions())

    for e in sim.units():
        px, py = _tile_px(e.x, e.y)
        col = COL_FRIENDLY if e.friendly else COL_ENEMY
        r = 6 if not e.stats.is_building else 10
        if e.stats.flying:
            cv2.drawMarker(panel, (px, py), col, cv2.MARKER_TRIANGLE_UP, r * 2, 2)
        else:
            cv2.circle(panel, (px, py), r, col, -1 if e.deployed else 1)
        _draw_hp_bar(panel, px, py - r - 6, e.hp_frac, 22)
        cv2.putText(panel, e.name[:7], (px - 20, py + r + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)

        if e.target_uid is not None:
            t = sim.entities.get(e.target_uid)
            if t is not None and t.alive:
                cv2.line(panel, (px, py), _tile_px(t.x, t.y), (90, 90, 90), 1)

    if trace is not None:
        _draw_placements(panel, trace)
    return panel


def _draw_observed(env) -> np.ndarray:
    """The tile grid as the policy receives it, noise included."""
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    _draw_field(panel)
    _draw_towers(panel, env.sim, env._reported_tower_hp)

    phantoms = getattr(env, "phantom_tiles", set())
    for ty in range(ARENA_ROWS):
        for tx in range(ARENA_COLS):
            troop = env.board.troops_in_arena[ty][tx]
            if troop is None:
                continue
            px, py = _tile_px(tx + 0.5, ty + 0.5)
            phantom = (tx, ty) in phantoms
            col = COL_GHOST if phantom else (
                COL_FRIENDLY if troop.color == "blue" else COL_ENEMY
            )
            cv2.rectangle(panel, (px - 9, py - 9), (px + 9, py + 9), col, 2)
            label = normalize_name(troop.name)[:7]
            cv2.putText(panel, f"{'?' if phantom else ''}{label}",
                        (px - 20, py + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, col, 1)

    # Units the detector dropped this frame, marked at their TRUE position.
    for (x, y) in getattr(env, "dropped_positions", []):
        px, py = _tile_px(x, y)
        cv2.drawMarker(panel, (px, py), (110, 110, 110),
                       cv2.MARKER_TILTED_CROSS, 14, 1)
    return panel


def _draw_sparkline(panel: np.ndarray, trace: RewardTrace, top: int) -> None:
    """Per-step reward over the recent past, as bars around a zero line.

    The SHAPE is the point: long flat stretches are the sparsity the shaping
    exists to break up.
    """
    h, pad = 66, 12
    mid = top + h // 2
    width = REWARD_W - 2 * pad
    cv2.putText(panel, f"per-step reward (last {trace.HISTORY})", (pad, top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_DIM, 1)
    cv2.rectangle(panel, (pad, top), (pad + width, top + h), (26, 24, 22), -1)
    cv2.line(panel, (pad, mid), (pad + width, mid), (70, 70, 70), 1)

    values = list(trace.history)
    if not values:
        return
    scale = max(abs(v) for v in values) or 1.0
    step_w = max(1, width // trace.HISTORY)
    for i, v in enumerate(values):
        x = pad + i * step_w
        if x > pad + width:
            break
        bar = int((v / scale) * (h // 2 - 2))
        if bar == 0:
            continue
        col = COL_GAIN if v > 0 else COL_LOSS
        cv2.rectangle(panel, (x, mid), (x + step_w - 1, mid - bar), col, -1)


def _draw_rewards(trace: RewardTrace) -> np.ndarray:
    """Running return, split by source."""
    panel = np.full((PANEL_H, REWARD_W, 3), COL_BG, dtype=np.uint8)
    pad = 12

    total_col = COL_GAIN if trace.total >= 0 else COL_LOSS
    cv2.putText(panel, "RETURN", (pad, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)
    cv2.putText(panel, f"{trace.total:+.2f}", (pad + 96, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, total_col, 2)

    last_col = COL_DIM if abs(trace.last) < 1e-9 else (
        COL_GAIN if trace.last > 0 else COL_LOSS
    )
    cv2.putText(panel, f"this step {trace.last:+.3f}", (pad, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, last_col, 1)
    cv2.putText(panel, f"silent {100 * trace.silent_fraction:.0f}% of steps",
                (pad, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_DIM, 1)

    y = 120
    cv2.putText(panel, "cumulative by source", (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_TEXT, 1)

    scale = max((abs(v) for v in trace.parts.values()), default=1.0) or 1.0
    bar_x, bar_w = pad + 148, REWARD_W - pad - 148 - pad
    mid = bar_x + bar_w // 2
    for name, value in trace.parts.items():
        y += 30
        live = name in trace.live
        label_col = COL_TEXT if live else COL_DIM
        cv2.putText(panel, name[:19], (pad, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, label_col, 1)
        cv2.putText(panel, f"{value:+7.2f}", (pad, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, label_col, 1)
        cv2.line(panel, (mid, y - 8), (mid, y + 8), (70, 70, 70), 1)
        w = int((value / scale) * (bar_w // 2))
        if w:
            c = COL_GAIN if value > 0 else COL_LOSS
            cv2.rectangle(panel, (mid, y - 6), (mid + w, y + 6), c, -1)
        if live:
            cv2.circle(panel, (bar_x - 8, y), 3, COL_TEXT, -1)

    _draw_sparkline(panel, trace, y + 44)
    return panel


def _draw_hud(width: int, env, trace: RewardTrace | None = None) -> np.ndarray:
    sim = env.sim
    hud = np.full((HUD_H, width, 3), COL_BG, dtype=np.uint8)

    mins, secs = divmod(int(sim.time), 60)
    versus = f"   vs {trace.opponent}" if trace and trace.opponent else ""
    left = [
        f"t {mins}:{secs:02d}  ({sim.phase()})   step {env.step_count}{versus}",
        f"crowns  you {sim.crowns[True]}  -  {sim.crowns[False]} them",
        f"hand: {'  '.join(env.hand)}",
    ]
    for i, line in enumerate(left):
        cv2.putText(hud, line, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)

    # Elixir pips, friendly side.
    bar_x, bar_y = width - 330, 20
    for i in range(10):
        filled = sim.elixir[True] >= i + 1
        x0 = bar_x + i * 26
        cv2.rectangle(hud, (x0, bar_y), (x0 + 22, bar_y + 16),
                      (200, 60, 200) if filled else (70, 45, 70), -1)
    cv2.putText(hud, f"{sim.elixir[True]:.1f}", (bar_x + 272, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)
    cv2.putText(hud, f"enemy elixir {sim.elixir[False]:.1f}",
                (bar_x, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_DIM, 1)

    if sim.finished:
        cv2.putText(hud, f"FINISHED: {sim.result}", (bar_x, bar_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
    return hud


def render(env, trace: RewardTrace | None = None) -> np.ndarray:
    """Full debug frame: HUD over TRUTH | OBSERVED [| REWARD]."""
    truth = _draw_truth(env.sim, trace)
    observed = _draw_observed(env)

    width = PANEL_W * 2 + GAP
    if trace is not None:
        width += GAP + REWARD_W
    body = np.full((PANEL_H, width, 3), COL_BG, dtype=np.uint8)
    body[:, :PANEL_W] = truth
    body[:, PANEL_W + GAP:PANEL_W * 2 + GAP] = observed
    if trace is not None:
        body[:, PANEL_W * 2 + GAP * 2:] = _draw_rewards(trace)

    hud = _draw_hud(width, env, trace)
    frame = np.vstack([hud, body])

    cv2.putText(frame, "TRUTH", (10, HUD_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    cv2.putText(frame, "OBSERVED (what the policy sees)",
                (PANEL_W + GAP + 10, HUD_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    if trace is not None:
        cv2.putText(frame, "REWARD", (PANEL_W * 2 + GAP * 2 + 10, HUD_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    return frame
