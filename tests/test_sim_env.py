"""Tests for SimEnv: schema parity with the real env, cadence, reward, noise.

The single most important property here is that a policy trained in the sim
reads the same bytes it will read on real hardware. Everything else is
secondary to that.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env.actions import Action
from src.env.observation import OBSERVATION_SHAPES, ObservationBuilder
from src.game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE
from src.game.classes import CARD_CLASSES
from src.sim import engine
from src.sim.env import (
    ELIXIR_CAP_PENALTY,
    STEP_PERIOD_SEC,
    TICKS_PER_STEP,
    TOWER_DESTROYED_REWARD,
    TOWER_LOST_PENALTY,
    Deck,
    ObservationNoise,
    SimEnv,
)


def clean_env(**kw) -> SimEnv:
    """Noise and stat randomization off - for tests about mechanics."""
    kw.setdefault("seed", 11)
    kw.setdefault("randomize_scale", 0.0)
    kw.setdefault("noise", ObservationNoise.off())
    return SimEnv(**kw)


class TestSchemaParity:
    def test_observation_matches_the_declared_schema(self):
        obs = clean_env().reset()
        assert set(obs) == set(OBSERVATION_SHAPES)
        for key, shape in OBSERVATION_SHAPES.items():
            assert np.asarray(obs[key]).shape == shape, key

    def test_flat_vector_is_the_real_encoder_output(self):
        env = clean_env()
        flat = ObservationBuilder.flatten(env.reset())
        expected = sum(int(np.prod(s)) for s in OBSERVATION_SHAPES.values())
        assert flat.shape == (expected,)
        assert flat.dtype == np.float32

    def test_hand_is_populated_from_the_deck(self):
        env = clean_env()
        obs = env.reset()
        assert obs["hand"].sum() == HAND_SIZE
        assert len(env.hand) == HAND_SIZE

    def test_placeable_mask_covers_only_the_friendly_half_at_start(self):
        obs = clean_env().reset()
        mask = obs["playable_mask"]
        assert mask[12].sum() == ARENA_COLS
        assert mask[3].sum() == 0

    def test_towers_start_at_full_health(self):
        obs = clean_env().reset()
        assert np.allclose(obs["tower_hp"], 1.0)


class TestCadence:
    def test_step_period_matches_the_measured_real_loop(self):
        """4 Hz, from ClashEnv.step_period_sec and 2098 recorded cycles.

        Training at a different rate teaches timings that do not survive
        deployment, so this is a contract and not a tuning knob.
        """
        assert STEP_PERIOD_SEC == 0.25
        assert TICKS_PER_STEP == 5

    def test_one_step_advances_one_period_of_game_time(self):
        env = clean_env()
        env.reset()
        _, _, _, info = env.step(Action.no_op())
        assert info["match_time"] == pytest.approx(STEP_PERIOD_SEC, abs=1e-6)

    def test_a_full_match_is_about_the_expected_number_of_steps(self):
        env = clean_env()
        env.reset()
        steps = 0
        done = False
        while not done and steps < 2000:
            _, _, done, info = env.step(Action.no_op())
            steps += 1
        assert done
        # Nothing is played, so it runs to the hard limit: 300s at 4 Hz.
        assert steps == pytest.approx(engine.MATCH_MAX_DURATION / STEP_PERIOD_SEC, rel=0.02)


class TestDeck:
    def test_playing_a_card_cycles_it_to_the_back(self):
        import random

        deck = Deck(tuple(CARD_CLASSES[:6]), random.Random(0))
        before = list(deck.hand)
        played = deck.play(0)
        assert played == before[0]
        assert deck.hand[0] != played
        assert deck.hand[1:] == before[1:]

    def test_a_played_card_returns_after_cycling_through(self):
        import random

        cards = tuple(CARD_CLASSES[:5])
        deck = Deck(cards, random.Random(0))
        played = deck.play(0)
        for _ in range(len(cards) - HAND_SIZE):
            deck.play(0)
        assert played in deck.hand

    def test_deck_rejects_cards_the_detector_cannot_see(self):
        import random

        with pytest.raises(ValueError, match="detector cannot recognise"):
            Deck(("wizard", "knight", "giant", "archer", "minion"), random.Random(0))

    def test_deck_must_be_larger_than_the_hand(self):
        import random

        with pytest.raises(ValueError, match="cycle"):
            Deck(tuple(CARD_CLASSES[:HAND_SIZE]), random.Random(0))


class TestActions:
    def test_placing_a_card_spends_elixir_and_puts_a_unit_on_the_field(self):
        env = clean_env()
        env.reset()
        env.sim.elixir[True] = 10.0
        before = len(env.sim.units(True))
        slot = 0
        _, _, _, info = env.step(Action(slot, 4, 12))
        assert info["action_ok"]
        assert len(env.sim.units(True)) > before
        assert env.sim.elixir[True] < 10.0

    def test_placing_on_the_enemy_half_is_rejected(self):
        env = clean_env()
        env.reset()
        env.sim.elixir[True] = 10.0
        _, _, _, info = env.step(Action(0, 4, 2))
        assert not info["action_ok"]
        assert info["action_reason"] == "tile_not_placeable"

    def test_placing_without_elixir_is_rejected(self):
        env = clean_env()
        env.reset()
        env.sim.elixir[True] = 0.0
        _, _, _, info = env.step(Action(0, 4, 12))
        assert not info["action_ok"]
        assert info["action_reason"] == "not_enough_elixir"

    def test_no_op_always_succeeds(self):
        env = clean_env()
        env.reset()
        _, _, _, info = env.step(Action.no_op())
        assert info["action_ok"]

    def test_a_rejected_action_does_not_consume_the_card(self):
        env = clean_env()
        env.reset()
        env.sim.elixir[True] = 0.0
        before = list(env.hand)
        env.step(Action(0, 4, 12))
        assert env.hand == before


class TestReward:
    """Reward is measured across a step, so these drive damage THROUGH a
    step rather than mutating between them - mutating in between is
    invisible to the env by design, and a test that did it would be
    asserting on an artefact."""

    @staticmethod
    def _step_until(env, predicate, limit=400):
        """Step until predicate(info) holds; return that step's reward."""
        for _ in range(limit):
            _, reward, done, info = env.step(Action.no_op())
            if predicate(info):
                return reward
            if done:
                break
        raise AssertionError("condition never occurred")

    def test_damaging_an_enemy_tower_pays(self):
        env = clean_env()
        env.reset()
        # A knight already past the river will reach the tower and hit it.
        env.sim._spawn("knight", True, 2.0, 4.2)
        before = env.sim.tower_hp_fractions()["enemy_left"]
        reward = self._step_until(
            env, lambda i: env.sim.tower_hp_fractions()["enemy_left"] < before
        )
        assert reward > 0

    def test_losing_tower_hp_costs(self):
        env = clean_env()
        env.reset()
        env.sim._spawn("knight", False, 2.0, 11.8)
        before = env.sim.tower_hp_fractions()["friendly_left"]
        reward = self._step_until(
            env, lambda i: env.sim.tower_hp_fractions()["friendly_left"] < before
        )
        assert reward < 0

    def test_destroying_a_tower_pays_a_discrete_bonus(self):
        env = clean_env()
        env.reset()
        tower = next(t for t in env.sim.towers(False) if t.tower_key == "enemy_left")
        tower.hp = 40.0  # one knight swing from falling
        env.sim._spawn("knight", True, 2.0, 4.2)
        reward = self._step_until(env, lambda i: i["crowns"][0] == 1)
        assert reward > TOWER_DESTROYED_REWARD

    def test_losing_a_tower_is_penalised(self):
        env = clean_env()
        env.reset()
        tower = next(t for t in env.sim.towers(True) if t.tower_key == "friendly_left")
        tower.hp = 40.0
        env.sim._spawn("knight", False, 2.0, 11.8)
        reward = self._step_until(env, lambda i: i["crowns"][1] == 1)
        assert reward < -TOWER_LOST_PENALTY

    def test_sitting_at_full_elixir_is_penalised(self):
        env = clean_env()
        env.reset()
        env.sim.elixir[True] = engine.MAX_ELIXIR
        _, reward, _, _ = env.step(Action.no_op())
        assert reward == pytest.approx(-ELIXIR_CAP_PENALTY)

    def test_spending_elixir_avoids_the_cap_penalty(self):
        env = clean_env()
        env.reset()
        env.sim.elixir[True] = engine.MAX_ELIXIR
        _, reward, _, info = env.step(Action(0, 4, 12))
        assert info["action_ok"]
        assert reward == pytest.approx(0.0)

    def test_an_illegal_action_is_penalised(self):
        env = clean_env()
        env.reset()
        env.sim.elixir[True] = 0.0
        _, reward, _, _ = env.step(Action(0, 4, 12))
        assert reward < 0

    def test_winning_pays_the_terminal_bonus(self):
        env = clean_env()
        env.reset()
        king = next(t for t in env.sim.towers(False) if t.tower_key == "enemy_king")
        king.hp = 40.0
        env.sim._spawn("knight", True, 4.5, 2.6)
        reward = self._step_until(env, lambda i: i["result"] == "win")
        assert reward > 10.0


class TestNoise:
    def test_noise_off_reports_the_field_exactly(self):
        env = clean_env()
        env.reset()
        # One unit, one tile: squads spread by less than a tile and would
        # collapse into a single cell, which is a property of the shared
        # board model rather than of the noise setting under test.
        env.sim._spawn("knight", True, 4.5, 12.5)
        obs = env.observe()
        assert obs["arena"].sum() == 1

    def test_noise_makes_observations_vary(self):
        noisy = SimEnv(seed=5, randomize_scale=0.0,
                       noise=ObservationNoise(drop_prob=0.5, position_jitter=0.4,
                                              false_positive_prob=0.0,
                                              tower_stale_prob=0.0))
        noisy.reset()
        for x in (2.5, 4.5, 6.5):
            noisy.sim._spawn("knight", True, x, 12.5)
        seen = {noisy.observe()["arena"].tobytes() for _ in range(30)}
        assert len(seen) > 1, "drop/jitter produced identical observations"

    def test_stale_tower_readings_hold_the_previous_value(self):
        env = SimEnv(seed=3, randomize_scale=0.0,
                     noise=ObservationNoise(drop_prob=0.0, position_jitter=0.0,
                                            false_positive_prob=0.0,
                                            tower_stale_prob=1.0))
        env.reset()
        tower = next(t for t in env.sim.towers(False) if t.tower_key == "enemy_left")
        tower.hp *= 0.4
        obs = env.step(Action.no_op())[0]
        idx = list(__import__("src.game.state", fromlist=["TOWER_KEYS"]).TOWER_KEYS).index("enemy_left")
        assert obs["tower_hp"][idx] == pytest.approx(1.0), (
            "occluded bar should hold its last value, not jump to the truth"
        )

    def test_randomization_changes_stats_between_episodes(self):
        env = SimEnv(seed=2, randomize_scale=1.0, noise=ObservationNoise.off())
        first = env.sim.unit_stats["knight"].hp
        env.reset()
        second = env.sim.unit_stats["knight"].hp
        assert first != second

    def test_randomization_off_keeps_the_table_exact(self):
        from src.sim.units import UNIT_STATS

        env = clean_env()
        assert env.sim.unit_stats["knight"].hp == UNIT_STATS["knight"].hp


class TestDeterminism:
    def test_same_seed_reproduces_the_episode(self):
        def rollout():
            env = SimEnv(seed=42, randomize_scale=1.0)
            env.reset()
            total = 0.0
            for i in range(60):
                a = Action(i % HAND_SIZE, 4, 12) if i % 5 == 0 else Action.no_op()
                _, r, done, _ = env.step(a)
                total += r
                if done:
                    break
            return round(total, 9), round(env.sim.time, 6)

        assert rollout() == rollout()

    def test_consecutive_episodes_differ(self):
        env = SimEnv(seed=9)
        first = env.sim.seed
        env.reset()
        assert env.sim.seed != first
