"""Behavioural tests for the simulation core.

RL will exploit a sim bug as readily as a real mechanic, and a policy built
on one transfers to nothing. These assert the behaviours that strategies are
actually built on - lane crossing, aggro, building-only targeting, king
activation, sudden death - rather than just that the code runs.
"""

from __future__ import annotations

import pytest

from src.sim import arena
from src.sim.engine import (
    MATCH_MAX_DURATION,
    REGULAR_TIME_END,
    TICK_DT,
    Simulation,
    can_attack,
)
from src.sim.units import UNIT_STATS, Target


def run_for(sim: Simulation, seconds: float) -> None:
    for _ in range(int(seconds / TICK_DT)):
        sim.tick()
        if sim.finished:
            return


def spawn(sim: Simulation, name: str, friendly: bool, x: float, y: float):
    e = sim._spawn(name, friendly, x, y)
    e.deploy_remaining = 0.0  # skip the deploy animation for test setup
    return e


class TestTargeting:
    def test_building_targeter_ignores_troops(self):
        """A Giant walking past a Musketeer is load-bearing for real decks."""
        sim = Simulation(seed=1)
        giant = spawn(sim, "giant", True, 4.5, 9.0)
        musket = spawn(sim, "musketeer", False, 4.6, 9.1)
        target = sim._acquire_target(giant)
        assert target is not None
        assert target.is_building
        assert target.uid != musket.uid

    def test_troop_diverts_to_a_nearby_enemy(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        goblin = spawn(sim, "goblin", False, 5.0, 9.0)
        assert sim._acquire_target(knight).uid == goblin.uid

    def test_troop_targets_a_tower_when_nothing_is_close(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        target = sim._acquire_target(knight)
        assert target.is_tower and not target.friendly

    def test_ground_only_unit_cannot_target_air(self):
        knight = UNIT_STATS["knight"]
        minion = UNIT_STATS["minion"]
        assert knight.targets is Target.GROUND
        assert minion.flying

        sim = Simulation(seed=1)
        k = spawn(sim, "knight", True, 4.5, 9.0)
        m = spawn(sim, "minion", False, 4.6, 9.0)
        assert not can_attack(k, m)
        assert sim._acquire_target(k).uid != m.uid

    def test_air_targeting_unit_can_hit_air(self):
        sim = Simulation(seed=1)
        musket = spawn(sim, "musketeer", True, 4.5, 9.0)
        minion = spawn(sim, "minion", False, 4.6, 9.0)
        assert can_attack(musket, minion)


class TestMovementAndLanes:
    def test_ground_unit_crosses_at_a_bridge(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 10.0)
        run_for(sim, 30.0)
        assert knight.y < arena.RIVER_Y, "knight never crossed the river"
        # It had to route via a bridge rather than walk over the water.
        assert min(abs(knight.x - bx) for bx in arena.BRIDGE_X) < 2.5

    def test_ground_unit_never_stands_in_open_water(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 10.0)
        for _ in range(int(30.0 / TICK_DT)):
            sim.tick()
            assert not arena.blocks_ground(knight.x, knight.y), (
                f"knight in water at ({knight.x:.2f}, {knight.y:.2f})"
            )
            if sim.finished:
                break

    def test_flying_unit_crosses_the_river_directly(self):
        sim = Simulation(seed=1)
        minion = spawn(sim, "minion", True, 4.5, 10.0)
        run_for(sim, 12.0)
        assert minion.y < arena.RIVER_Y
        # Straight line: it should not have detoured toward a bridge.
        assert abs(minion.x - 4.5) < 2.0

    def test_unit_advances_toward_the_enemy_side(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 12.0)
        start_y = knight.y
        run_for(sim, 5.0)
        assert knight.y < start_y


class TestCombat:
    def test_a_unit_damages_the_tower_it_reaches(self):
        sim = Simulation(seed=1)
        spawn(sim, "knight", True, 2.0, 5.0)  # already past the river
        before = sim.tower_hp_fractions()["enemy_left"]
        run_for(sim, 20.0)
        assert sim.tower_hp_fractions()["enemy_left"] < before

    def test_towers_shoot_back(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 2.0, 4.0)
        run_for(sim, 10.0)
        assert knight.hp < knight.max_hp

    def test_deploy_time_delays_action(self):
        sim = Simulation(seed=1)
        e = sim._spawn("knight", True, 2.0, 5.0)
        assert e.deploy_remaining > 0
        start = (e.x, e.y)
        sim.tick()
        assert (e.x, e.y) == start, "unit acted before finishing deployment"

    def test_dead_units_are_removed(self):
        sim = Simulation(seed=1)
        weak = spawn(sim, "speargoblin", True, 2.0, 3.6)
        uid = weak.uid
        run_for(sim, 25.0)
        assert uid not in sim.entities


class TestSpells:
    def test_spell_damages_enemies_in_radius(self):
        sim = Simulation(seed=1)
        target = spawn(sim, "knight", False, 4.5, 9.0)
        sim.elixir[True] = 10.0
        before = target.hp
        sim._cast_spell(True, "fireball", 4.5, 9.0)
        assert target.hp < before

    def test_spell_spares_friendly_units(self):
        sim = Simulation(seed=1)
        friend = spawn(sim, "knight", True, 4.5, 9.0)
        sim._cast_spell(True, "fireball", 4.5, 9.0)
        assert friend.hp == friend.max_hp

    def test_spell_misses_outside_radius(self):
        sim = Simulation(seed=1)
        far = spawn(sim, "knight", False, 8.0, 9.0)
        sim._cast_spell(True, "fireball", 1.0, 9.0)
        assert far.hp == far.max_hp

    def test_spells_hit_towers_for_reduced_damage(self):
        sim = Simulation(seed=1)
        tower = next(t for t in sim.towers(False) if t.tower_key == "enemy_left")
        before = tower.hp
        sim._cast_spell(True, "fireball", tower.x, tower.y)
        dealt = before - tower.hp
        full = 325.0
        assert 0 < dealt < full, "tower took full spell damage"


class TestBuildings:
    def test_spawner_produces_units(self):
        sim = Simulation(seed=1)
        spawn(sim, "goblinhut", True, 4.5, 11.0)
        before = len(sim.units(True))
        run_for(sim, 12.0)
        assert len(sim.units(True)) > before

    def test_building_expires_at_end_of_lifetime(self):
        sim = Simulation(seed=1)
        hut = spawn(sim, "goblinhut", True, 4.5, 11.0)
        uid = hut.uid
        run_for(sim, UNIT_STATS["goblinhut"].lifetime + 2.0)
        assert uid not in sim.entities

    def test_goblin_cage_releases_a_brawler_on_death(self):
        """The spawn-only unit that motivates the two class lists."""
        sim = Simulation(seed=1)
        cage = spawn(sim, "goblincage", True, 4.5, 11.0)
        assert not any(u.name == "goblinbrawler" for u in sim.units(True))
        sim._damage(cage, cage.hp)
        assert any(u.name == "goblinbrawler" for u in sim.units(True))


class TestTowersAndCrowns:
    def test_destroying_a_princess_scores_one_crown(self):
        sim = Simulation(seed=1)
        tower = next(t for t in sim.towers(False) if t.tower_key == "enemy_left")
        sim._damage(tower, tower.hp)
        assert sim.crowns[True] == 1
        assert "enemy_left" in sim.destroyed_towers

    def test_king_starts_inactive_and_wakes_when_a_princess_falls(self):
        sim = Simulation(seed=1)
        king = next(t for t in sim.towers(False) if t.tower_key == "enemy_king")
        assert not king.active
        princess = next(t for t in sim.towers(False) if t.tower_key == "enemy_right")
        sim._damage(princess, princess.hp)
        assert king.active

    def test_king_wakes_when_damaged(self):
        sim = Simulation(seed=1)
        king = next(t for t in sim.towers(False) if t.tower_key == "enemy_king")
        assert not king.active
        sim._damage(king, 200.0)
        assert king.active

    def test_destroying_the_king_ends_the_match_at_three_crowns(self):
        sim = Simulation(seed=1)
        king = next(t for t in sim.towers(False) if t.tower_key == "enemy_king")
        sim._damage(king, king.hp)
        assert sim.finished
        assert sim.result == "win"
        assert sim.crowns[True] == 3

    def test_losing_your_own_king_is_a_loss(self):
        sim = Simulation(seed=1)
        king = next(t for t in sim.towers(True) if t.tower_key == "friendly_king")
        sim._damage(king, king.hp)
        assert sim.finished and sim.result == "loss"


class TestMatchFlow:
    def test_match_continues_past_regular_time_when_crowns_are_level(self):
        sim = Simulation(seed=1)
        sim.time = REGULAR_TIME_END + 1.0
        sim.tick()
        assert not sim.finished

    def test_regular_time_ends_the_match_when_ahead(self):
        sim = Simulation(seed=1)
        sim.crowns[True] = 1
        sim.time = REGULAR_TIME_END - TICK_DT / 2
        sim.tick()
        assert sim.finished and sim.result == "win"

    def test_overtime_is_sudden_death(self):
        sim = Simulation(seed=1)
        sim.time = REGULAR_TIME_END + 10.0
        tower = next(t for t in sim.towers(False) if t.tower_key == "enemy_left")
        sim._damage(tower, tower.hp)
        assert sim.finished and sim.result == "win"

    def test_match_ends_at_the_hard_time_limit(self):
        sim = Simulation(seed=1)
        sim.time = MATCH_MAX_DURATION - TICK_DT / 2
        sim.tick()
        assert sim.finished

    def test_level_match_at_time_is_broken_by_lowest_tower(self):
        sim = Simulation(seed=1)
        enemy = next(t for t in sim.towers(False) if t.tower_key == "enemy_left")
        enemy.hp = enemy.max_hp * 0.10
        sim.time = MATCH_MAX_DURATION
        sim.tick()
        assert sim.finished and sim.result == "win"

    def test_untouched_match_is_a_draw(self):
        sim = Simulation(seed=1)
        sim.time = MATCH_MAX_DURATION
        sim.tick()
        assert sim.finished and sim.result == "draw"


class TestElixir:
    @pytest.mark.parametrize(
        "time_s,expected_phase",
        [(0.0, "normal"), (130.0, "double"),
         (200.0, "overtime_double"), (260.0, "overtime_triple")],
    )
    def test_phases_match_the_real_state_machine(self, time_s, expected_phase):
        sim = Simulation(seed=1)
        sim.time = time_s
        assert sim.phase() == expected_phase

    def test_elixir_regenerates_and_caps(self):
        sim = Simulation(seed=1)
        sim.elixir[True] = 0.0
        run_for(sim, 3.0)
        assert 0.9 < sim.elixir[True] < 1.2  # ~1 per 2.8s
        run_for(sim, 60.0)
        assert sim.elixir[True] == pytest.approx(10.0)

    def test_double_elixir_is_faster(self):
        sim = Simulation(seed=1)
        sim.time = 130.0
        sim.elixir[True] = 0.0
        run_for(sim, 2.8)
        assert sim.elixir[True] > 1.5


class TestPlacement:
    def test_own_half_is_placeable(self):
        sim = Simulation(seed=1)
        assert sim.is_placeable(True, 4, 12)
        assert not sim.is_placeable(True, 4, 3)

    def test_destroying_a_tower_opens_that_lane(self):
        sim = Simulation(seed=1)
        assert not sim.is_placeable(True, 1, 3)
        tower = next(t for t in sim.towers(False) if t.tower_key == "enemy_left")
        sim._damage(tower, tower.hp)
        assert sim.is_placeable(True, 1, 3)
        assert not sim.is_placeable(True, 7, 3), "wrong lane opened"

    def test_deploy_costs_elixir(self):
        sim = Simulation(seed=1)
        sim.elixir[True] = 10.0
        assert sim.deploy(True, "knight", 4, 12)
        assert sim.elixir[True] == pytest.approx(7.0)

    def test_riverbank_row_is_never_placeable_for_troops(self):
        """Row 7 stays shut even once its lane opens."""
        sim = Simulation(seed=1)
        tower = next(t for t in sim.towers(False) if t.tower_key == "enemy_left")
        sim._damage(tower, tower.hp)
        assert sim.is_placeable(True, 1, 6)
        assert not sim.is_placeable(True, 1, 7)
        assert sim.is_placeable(True, 1, 7, name="arrows"), "spells go anywhere"

    def test_enemy_king_activation_grants_no_ground(self):
        """King activation makes it shoot; it does not open the enemy half.

        The board's mask used to grant this and the simulator never did, so
        the policy aimed at tiles the env then refused. Pinned on both sides
        by ``test_mask_matches_the_env``.
        """
        sim = Simulation(seed=1)
        king = next(t for t in sim.towers(False) if t.tower_key == "enemy_king")
        sim._damage(king, king.hp * 0.5)
        assert not sim.is_placeable(True, 1, 3)
        assert not sim.is_placeable(True, 7, 3)


class TestMaskAgreesWithEnv:
    """The observation's ``playable_mask`` and the env's acceptance are two
    reads of ONE rule. When they drifted, 60% of a trained policy's
    placements were refused and it retried the same tile every step.

    Noise is OFF here. With it on the mask is *allowed* to lag - an occluded
    tower bar holds its previous value, exactly as the vision pipeline does -
    and tower state is only re-read on a step, so these settle the sim with a
    no-op before comparing.
    """

    @staticmethod
    def _settle(env) -> None:
        from src.env.actions import Action

        env.step(Action.no_op())

    @staticmethod
    def _check(env) -> None:
        from src.game.board import ARENA_COLS, ARENA_ROWS

        obs = env.observe()
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                masked = obs["playable_mask"][y][x] > 0.5
                allowed = env.sim.is_placeable(True, x, y, name="knight")
                assert masked == allowed, (
                    f"tile ({x}, {y}): mask says {masked}, env says {allowed}"
                )

    def test_mask_matches_the_env(self):
        from src.sim.env import ObservationNoise, SimEnv

        env = SimEnv(seed=3, noise=ObservationNoise.off())
        env.reset()
        self._check(env)

    def test_mask_matches_the_env_with_a_damaged_king(self):
        from src.sim.env import ObservationNoise, SimEnv

        env = SimEnv(seed=3, noise=ObservationNoise.off())
        env.reset()
        king = next(t for t in env.sim.towers(False) if t.tower_key == "enemy_king")
        env.sim._damage(king, king.hp * 0.5)
        self._settle(env)
        self._check(env)

    def test_mask_matches_the_env_with_a_lane_open(self):
        from src.sim.env import ObservationNoise, SimEnv

        env = SimEnv(seed=3, noise=ObservationNoise.off())
        env.reset()
        tower = next(
            t for t in env.sim.towers(False) if t.tower_key == "enemy_left"
        )
        env.sim._damage(tower, tower.hp)
        self._settle(env)
        self._check(env)

    def test_deploy_refused_without_elixir(self):
        sim = Simulation(seed=1)
        sim.elixir[True] = 1.0
        assert not sim.deploy(True, "giant", 4, 12)

    def test_squad_cards_spawn_their_whole_count(self):
        sim = Simulation(seed=1)
        sim.elixir[True] = 10.0
        sim.deploy(True, "goblin", 4, 12)
        assert len(sim.units(True)) == UNIT_STATS["goblin"].count


class TestDeterminism:
    def test_same_seed_gives_the_same_match(self):
        def play(seed: int):
            sim = Simulation(seed=seed)
            sim.elixir[True] = 10.0
            sim.deploy(True, "giant", 4, 12)
            run_for(sim, 40.0)
            return sim.tower_hp_fractions(), sim.time

        assert play(3) == play(3)
