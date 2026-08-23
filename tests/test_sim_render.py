"""Smoke tests for the simulator's visual debugger.

The renderer is a debugging tool, so it is not worth asserting pixels. What
IS worth asserting is that it never raises and never lies about the noise
model - a debug view that silently mislabels dropped units as phantoms is
worse than no debug view, because it sends you looking for the wrong bug.
"""

from __future__ import annotations

import numpy as np

from src.env.actions import Action
from src.sim.env import ObservationNoise, SimEnv
from src.sim.render import HUD_H, PANEL_H, PANEL_W, render


def test_render_produces_a_populated_frame():
    env = SimEnv(seed=4, randomize_scale=0.0, noise=ObservationNoise.off())
    env.reset()
    env.sim._spawn("knight", True, 4.5, 10.0)
    env.sim._spawn("minion", False, 4.5, 6.0)
    env.step(Action.no_op())

    frame = render(env)
    assert frame.shape[0] == HUD_H + PANEL_H
    assert frame.dtype == np.uint8
    assert frame.std() > 10, "frame looks blank"


def test_render_survives_a_finished_match():
    env = SimEnv(seed=4, randomize_scale=0.0, noise=ObservationNoise.off())
    env.reset()
    king = next(t for t in env.sim.towers(False) if t.tower_key == "enemy_king")
    env.sim._damage(king, king.hp)
    env.step(Action.no_op())
    assert env.sim.finished
    render(env)  # destroyed towers take a different draw path


def test_noise_events_are_recorded_not_inferred():
    """Jitter moves a real unit to a neighbouring tile. If phantoms were
    inferred by comparing the board against truth positions, every jittered
    unit would be mislabelled as a false positive."""
    env = SimEnv(
        seed=4, randomize_scale=0.0,
        noise=ObservationNoise(drop_prob=0.0, position_jitter=0.9,
                               false_positive_prob=0.0, tower_stale_prob=0.0),
    )
    env.reset()
    for x in (2.0, 4.5, 7.0):
        env.sim._spawn("knight", True, x, 10.0)
    env.step(Action.no_op())

    assert env.phantom_tiles == set(), "jitter was reported as a phantom"
    assert env.dropped_positions == []


def test_dropped_units_are_reported():
    env = SimEnv(
        seed=4, randomize_scale=0.0,
        noise=ObservationNoise(drop_prob=1.0, position_jitter=0.0,
                               false_positive_prob=0.0, tower_stale_prob=0.0),
    )
    env.reset()
    env.sim._spawn("knight", True, 4.5, 10.0)
    env.step(Action.no_op())
    assert len(env.dropped_positions) == 1
    assert env.observe()["arena"].sum() == 0


def test_phantoms_are_reported():
    env = SimEnv(
        seed=4, randomize_scale=0.0,
        noise=ObservationNoise(drop_prob=0.0, position_jitter=0.0,
                               false_positive_prob=1.0, tower_stale_prob=0.0),
    )
    env.reset()
    env.step(Action.no_op())
    assert len(env.phantom_tiles) == 1
    assert env.observe()["arena"].sum() >= 1
