"""Visual debugger for the simulator.

Draws two panels side by side:

- TRUTH: where entities actually are, at full float precision.
- OBSERVED: the tile grid the policy is handed, after detection dropout,
  position jitter, phantom units and stale tower bars.

The pair is the point. Sim fidelity and perception noise are the two things
that decide whether a policy transfers, and both are invisible in a reward
curve. Seeing a unit vanish from the right panel while it is plainly there on
the left is the fastest way to understand what the policy is working with.

Rendering only - it never advances the sim, so it cannot perturb a run.
"""

from __future__ import annotations

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
        r = int(spec.radius * TILE)
        col = COL_FRIENDLY if spec.friendly else COL_ENEMY
        if frac <= 0.0:
            cv2.rectangle(panel, (px - r, py - r), (px + r, py + r), (60, 60, 60), -1)
            cv2.line(panel, (px - r, py - r), (px + r, py + r), (30, 30, 30), 2)
            cv2.line(panel, (px - r, py + r), (px + r, py - r), (30, 30, 30), 2)
            continue
        cv2.rectangle(panel, (px - r, py - r), (px + r, py + r), col, 2)
        _draw_hp_bar(panel, px, py + r + 8, frac, r * 2)
        if spec.is_king:
            entity = next(
                (e for e in sim.entities.values() if e.tower_key == spec.key), None
            )
            if entity is not None and not entity.active:
                cv2.putText(panel, "zzz", (px - 14, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DIM, 1)


def _draw_truth(sim: Simulation) -> np.ndarray:
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


def _draw_hud(width: int, env) -> np.ndarray:
    sim = env.sim
    hud = np.full((HUD_H, width, 3), COL_BG, dtype=np.uint8)

    mins, secs = divmod(int(sim.time), 60)
    left = [
        f"t {mins}:{secs:02d}  ({sim.phase()})   step {env.step_count}",
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


def render(env) -> np.ndarray:
    """Full debug frame: HUD over TRUTH | OBSERVED."""
    truth = _draw_truth(env.sim)
    observed = _draw_observed(env)

    width = PANEL_W * 2 + GAP
    body = np.full((PANEL_H, width, 3), COL_BG, dtype=np.uint8)
    body[:, :PANEL_W] = truth
    body[:, PANEL_W + GAP:] = observed

    hud = _draw_hud(width, env)
    frame = np.vstack([hud, body])

    cv2.putText(frame, "TRUTH", (10, HUD_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    cv2.putText(frame, "OBSERVED (what the policy sees)",
                (PANEL_W + GAP + 10, HUD_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1)
    return frame
