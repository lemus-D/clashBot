"""Card and tower LEVEL variation, and the fact that it stays hidden.

Ladder play does not match players exactly - an opponent is often a level or
two off, and your tower level moves independently of your card levels. The
simulator samples that per episode.

The property that matters most is the negative one: level must NOT reach the
observation. The detector reports "knight" with no level attached, so a policy
that could see levels in training would learn something it cannot use in
deployment. Several tests here exist only to assert that absence.
"""

from __future__ import annotations

import random

import pytest

from src.env.actions import Action
from src.env.observation import OBSERVATION_SHAPES, schema_hash
from src.sim.env import LevelSpread, ObservationNoise, SimEnv
from src.sim.units import (
    LEVELS,
    STANDARD_LEVEL,
    stats_at_level,
    tower_combat,
)


class TestStatsByLevel:
    def test_higher_level_means_more_hp_and_damage(self):
        lo, _ = stats_at_level(min(LEVELS))
        hi, _ = stats_at_level(max(LEVELS))
        for name in ("knight", "musketeer", "giant"):
            assert hi[name].hp > lo[name].hp
            assert hi[name].damage > lo[name].damage

    def test_non_scaling_stats_do_not_move(self):
        """Only HP and damage scale with level in Clash Royale."""
        lo, _ = stats_at_level(min(LEVELS))
        hi, _ = stats_at_level(max(LEVELS))
        for name in lo:
            assert lo[name].speed == hi[name].speed, name
            assert lo[name].attack_range == hi[name].attack_range, name
            assert lo[name].hit_speed == hi[name].hit_speed, name
            assert lo[name].count == hi[name].count, name

    def test_level_is_clamped_to_the_emitted_band(self):
        below, _ = stats_at_level(min(LEVELS) - 5)
        at, _ = stats_at_level(min(LEVELS))
        assert below["knight"].hp == at["knight"].hp

    def test_towers_scale_too(self):
        lo = tower_combat("princess", min(LEVELS))
        hi = tower_combat("princess", max(LEVELS))
        assert hi["hp"] > lo["hp"] and hi["damage"] > lo["damage"]
        assert hi["attack_range"] == lo["attack_range"]

    def test_spells_scale_too(self):
        _, lo = stats_at_level(min(LEVELS))
        _, hi = stats_at_level(max(LEVELS))
        assert hi["fireball"][0] > lo["fireball"][0]
        assert hi["fireball"][1] == lo["fireball"][1]  # radius is fixed

    def test_standard_level_is_in_the_band(self):
        assert min(LEVELS) <= STANDARD_LEVEL <= max(LEVELS)


class TestSampling:
    def test_samples_stay_inside_the_band(self):
        spread = LevelSpread()
        rng = random.Random(0)
        for _ in range(500):
            for v in spread.sample(rng).values():
                assert min(LEVELS) <= v <= max(LEVELS)

    def test_spread_is_respected(self):
        spread = LevelSpread(base=11, troop_spread=1, tower_spread=0)
        rng = random.Random(0)
        for _ in range(200):
            s = spread.sample(rng)
            assert abs(s["friendly"] - 11) <= 1
            assert s["friendly_tower"] == s["friendly"]

    def test_sides_are_sampled_independently(self):
        spread = LevelSpread()
        rng = random.Random(1)
        seen = {(s["friendly"], s["enemy"]) for s in
                (spread.sample(rng) for _ in range(200))}
        assert any(f != e for f, e in seen), "both sides always equal"

    def test_towers_can_differ_from_troops(self):
        spread = LevelSpread()
        rng = random.Random(2)
        assert any(
            s["friendly_tower"] != s["friendly"]
            for s in (spread.sample(rng) for _ in range(200))
        )

    def test_off_pins_everything_to_base(self):
        s = LevelSpread.off().sample(random.Random(0))
        assert set(s.values()) == {STANDARD_LEVEL}


class TestEnvIntegration:
    def test_levels_vary_across_episodes(self):
        env = SimEnv(seed=5, noise=ObservationNoise.off())
        seen = set()
        for _ in range(30):
            env.reset()
            seen.add(tuple(sorted(env.episode_levels.items())))
        assert len(seen) > 1

    def test_each_side_gets_its_own_stat_table(self):
        env = SimEnv(seed=5, randomize_scale=0.0,
                     levels=LevelSpread(base=11, troop_spread=2, tower_spread=0),
                     noise=ObservationNoise.off())
        for _ in range(30):
            env.reset()
            if env.episode_levels["friendly"] != env.episode_levels["enemy"]:
                f = env.sim._stats[True]["knight"].hp
                e = env.sim._stats[False]["knight"].hp
                assert f != e
                return
        pytest.skip("no episode sampled differing side levels")

    def test_tower_hp_follows_the_tower_level(self):
        env = SimEnv(seed=5, randomize_scale=0.0, noise=ObservationNoise.off(),
                     levels=LevelSpread(base=11, troop_spread=0, tower_spread=2))
        for _ in range(30):
            env.reset()
            lv = env.episode_levels
            if lv["friendly_tower"] != lv["enemy_tower"]:
                f = next(t for t in env.sim.towers(True)
                         if t.tower_key == "friendly_left").max_hp
                e = next(t for t in env.sim.towers(False)
                         if t.tower_key == "enemy_left").max_hp
                assert (f > e) == (lv["friendly_tower"] > lv["enemy_tower"])
                return
        pytest.skip("no episode sampled differing tower levels")

    def test_levels_off_reproduces_standard_stats(self):
        env = SimEnv(seed=5, randomize_scale=0.0, levels=LevelSpread.off(),
                     noise=ObservationNoise.off())
        env.reset()
        std, _ = stats_at_level(STANDARD_LEVEL)
        assert env.sim._stats[True]["knight"].hp == std["knight"].hp
        assert env.sim._stats[False]["knight"].hp == std["knight"].hp

    def test_episodes_stay_deterministic_under_a_seed(self):
        def rollout():
            env = SimEnv(seed=77)
            env.reset()
            for _ in range(40):
                env.step(Action.no_op())
            return dict(env.episode_levels), round(env.sim.time, 6)

        assert rollout() == rollout()


class TestLevelStaysHidden:
    """The detector cannot read levels, so neither can the policy."""

    def test_observation_has_no_level_field(self):
        env = SimEnv(seed=5)
        obs = env.reset()
        assert set(obs) == set(OBSERVATION_SHAPES)
        assert not any("level" in k for k in obs)

    def test_level_variation_does_not_change_the_schema(self):
        """If levels touched the schema they would be visible to the policy
        and would also invalidate checkpoints every episode."""
        before = schema_hash()
        env = SimEnv(seed=5, levels=LevelSpread(base=11, troop_spread=2))
        for _ in range(10):
            env.reset()
            assert schema_hash() == before

    def test_two_episodes_at_different_levels_look_identical_at_reset(self):
        """Same board, same hand -> same observation, whatever the levels.

        Tower HP is a normalised FILL FRACTION, so a level-14 tower and a
        level-9 tower both read 1.000 at full health. That is what makes
        level genuinely unobservable rather than merely unlabelled.
        """
        env = SimEnv(seed=5, randomize_scale=0.0, noise=ObservationNoise.off(),
                     levels=LevelSpread(base=11, troop_spread=2))
        seen = {}
        for _ in range(30):
            obs = env.reset()
            key = env.episode_levels["friendly"]
            seen.setdefault(key, []).append(obs["tower_hp"].tobytes())
        for level, vals in seen.items():
            assert len(set(vals)) == 1, level
        distinct_levels = list(seen)
        if len(distinct_levels) > 1:
            a, b = distinct_levels[:2]
            assert seen[a][0] == seen[b][0], (
                "tower_hp differs by level - level has leaked into the "
                "observation via absolute HP instead of fill fraction"
            )

    def test_levels_are_reported_in_info_for_debugging(self):
        env = SimEnv(seed=5)
        env.reset()
        _, _, _, info = env.step(Action.no_op())
        assert set(info["levels"]) == {
            "friendly", "enemy", "friendly_tower", "enemy_tower",
        }
