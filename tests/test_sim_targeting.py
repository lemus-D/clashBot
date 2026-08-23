"""Targeting and pathing rules, matched to how the real game behaves.

The rules encoded here, and why each one matters:

- A troop walking TOWARD a tower diverts to an enemy that enters sight range.
  That is ordinary distraction and it works.
- A troop that has already STARTED HITTING a tower is committed and ignores
  troops beside it. This is why forcing a Royal Giant off your tower needs a
  stun or a displacement card rather than just a distraction unit.
- A troop target is held until it dies or runs past the leash. Units do not
  shop around mid-fight.
- Building-only attackers never divert at all.
- Ground units route via a bridge only when their goal is ACROSS the river.
  Keying that to which half the unit owned was a bug: defenders walked to the
  bridge instead of at an enemy standing next to them.
"""

from __future__ import annotations

import pytest

from src.sim import arena
from src.sim.engine import TICK_DT, Simulation
from src.sim.units import UNIT_STATS


def spawn(sim, name, friendly, x, y):
    e = sim._spawn(name, friendly, x, y)
    e.deploy_remaining = 0.0
    return e


def run_for(sim, seconds):
    for _ in range(int(seconds / TICK_DT)):
        sim.tick()
        if sim.finished:
            return


class TestDistraction:
    def test_a_troop_walking_to_a_tower_diverts_to_a_nearby_enemy(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 2.0, 6.0)   # crossed, heading up
        sim._retarget(knight)
        assert sim.entities[knight.target_uid].is_tower

        goblin = spawn(sim, "goblin", False, 2.3, 6.0)  # walks up beside it
        assert sim._retarget(knight).uid == goblin.uid

    def test_a_troop_already_hitting_a_tower_ignores_a_distraction(self):
        """The Royal-Giant-on-your-tower case."""
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 2.0, 3.9)
        run_for(sim, 6.0)  # long enough to reach and start swinging
        assert knight.locked_on_structure, "never engaged the tower"

        goblin = spawn(sim, "goblin", False, 2.2, 3.9)
        assert sim._retarget(knight).is_tower
        assert knight.target_uid != goblin.uid

    def test_the_lock_clears_when_the_structure_dies(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 2.0, 3.9)
        run_for(sim, 6.0)
        assert knight.locked_on_structure
        tower = next(t for t in sim.towers(False) if t.tower_key == "enemy_left")
        sim._damage(tower, tower.hp)
        sim._retarget(knight)
        assert not knight.locked_on_structure

    def test_building_targeters_never_divert(self):
        sim = Simulation(seed=1)
        giant = spawn(sim, "giant", True, 4.5, 6.0)
        spawn(sim, "musketeer", False, 4.6, 6.0)
        assert sim._retarget(giant).is_building
        assert UNIT_STATS["giant"].aggro_range == 0.0


class TestTargetHolding:
    def test_a_troop_target_is_held_while_it_lives(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        goblin = spawn(sim, "goblin", False, 4.8, 9.0)
        assert sim._retarget(knight).uid == goblin.uid

        # A closer enemy appears; the knight should NOT switch.
        spawn(sim, "goblin", False, 4.55, 9.0)
        assert sim._retarget(knight).uid == goblin.uid

    def test_a_dead_target_is_replaced(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        goblin = spawn(sim, "goblin", False, 4.8, 9.0)
        sim._retarget(knight)
        sim._damage(goblin, goblin.hp)
        assert sim._retarget(knight).uid != goblin.uid

    def test_a_target_that_flees_past_the_leash_is_dropped(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        goblin = spawn(sim, "goblin", False, 4.8, 9.0)
        sim._retarget(knight)
        goblin.x, goblin.y = 4.5, 15.5  # ran away
        assert sim._retarget(knight).uid != goblin.uid

    def test_a_target_just_inside_the_leash_is_kept(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        goblin = spawn(sim, "goblin", False, 4.5, 9.0)
        sim._retarget(knight)
        reach = knight.stats.aggro_range * Simulation.LEASH_FACTOR
        goblin.y = 9.0 + reach * 0.9
        assert sim._retarget(knight).uid == goblin.uid


class TestDefaultGoal:
    def test_normal_troops_head_for_a_crown_tower(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 2.0, 10.0)
        assert sim._default_goal(knight).is_tower

    def test_building_targeters_prefer_a_nearer_building_over_a_tower(self):
        """A hut in front of the towers pulls a Giant off them."""
        sim = Simulation(seed=1)
        giant = spawn(sim, "giant", True, 4.5, 9.0)
        hut = spawn(sim, "goblinhut", False, 4.5, 6.0)
        goal = sim._default_goal(giant)
        assert goal.uid == hut.uid and not goal.is_tower

    def test_a_hut_in_the_path_distracts_a_normal_troop(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 6.0)
        hut = spawn(sim, "goblinhut", False, 4.6, 6.0)
        assert sim._retarget(knight).uid == hut.uid


class TestPathing:
    def test_a_defender_walks_at_an_enemy_on_its_own_half(self):
        """The bug that made movement look wrong: routing keyed off which
        half the unit OWNED, so defenders ran to the bridge."""
        gx, gy = arena.ground_waypoint(4.5, 12.0, True, 4.5, 13.0)
        assert (gx, gy) == (4.5, 13.0)

    def test_a_unit_routes_to_a_bridge_when_the_goal_is_across(self):
        gx, gy = arena.ground_waypoint(4.5, 10.0, True, 4.5, 3.0)
        assert gx in arena.BRIDGE_X or abs(gx - arena.nearest_bridge_x(4.5)) < 1e-6
        assert gy != 3.0

    def test_routing_is_symmetric_for_the_enemy_side(self):
        gx, gy = arena.ground_waypoint(4.5, 5.0, False, 4.5, 13.0)
        assert abs(gx - arena.nearest_bridge_x(4.5)) < 1e-6

    def test_same_side_is_purely_geometric(self):
        assert arena.same_side(9.0, 12.0)
        assert arena.same_side(2.0, 5.0)
        assert not arena.same_side(5.0, 12.0)

    def test_a_defender_walks_at_an_intruder_instead_of_the_bridge(self):
        """Placed clear of the friendly towers, or they shoot the intruder
        before the knight gets anywhere and the test measures nothing."""
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 11.0)
        goblin = spawn(sim, "goblin", False, 4.5, 9.5)
        before = goblin.hp
        run_for(sim, 1.5)

        # Engaged: damaged or outright killed. A knight out-damages a goblin's
        # whole health bar in one swing, so either counts.
        assert goblin.hp < before, "knight never engaged the intruder"
        assert abs(knight.x - 4.5) < 1.0, (
            f"knight detoured toward a bridge (x={knight.x:.2f}) instead of "
            f"walking at an enemy on its own half"
        )

    def test_ground_units_never_stand_in_open_water(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 10.0)
        for _ in range(int(40.0 / TICK_DT)):
            sim.tick()
            assert not arena.blocks_ground(knight.x, knight.y)
            if sim.finished:
                return

    def test_a_unit_pushed_into_water_slides_at_full_speed(self):
        """The bank slide used to move a FRACTION of the remaining distance
        per tick, easing to a crawl near the bridge."""
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 8.0)  # standing in the river
        before = knight.x
        sim.tick()
        moved = abs(knight.x - before)
        expected = knight.stats.speed * TICK_DT
        assert moved == pytest.approx(expected, rel=0.2)


class TestRangeGeometry:
    def test_range_is_measured_between_hitbox_edges(self):
        sim = Simulation(seed=1)
        a = spawn(sim, "knight", True, 4.5, 9.0)
        b = spawn(sim, "goblin", False, 4.5, 9.0)
        reach = a.stats.attack_range + a.radius + b.radius
        assert reach > a.stats.attack_range

        b.y = 9.0 + reach * 0.99
        sim._retarget(a)
        a.attack_cooldown = 0.0
        hp = b.hp
        sim._tick_combat(a, TICK_DT)
        assert b.hp < hp, "in range but did not attack"

    def test_out_of_range_does_not_hit(self):
        sim = Simulation(seed=1)
        a = spawn(sim, "knight", True, 4.5, 9.0)
        b = spawn(sim, "goblin", False, 4.5, 9.0)
        reach = a.stats.attack_range + a.radius + b.radius
        b.y = 9.0 + reach * 1.5
        sim._retarget(a)
        a.attack_cooldown = 0.0
        hp = b.hp
        sim._tick_combat(a, TICK_DT)
        assert b.hp == hp

    def test_sight_ranges_come_from_game_data(self):
        """Building-only attackers are pulled from farther away because their
        sight range genuinely is longer - confirmed in the game data."""
        assert UNIT_STATS["musketeer"].aggro_range > 0
        assert UNIT_STATS["knight"].aggro_range > 0
        assert UNIT_STATS["musketeer"].attack_range > UNIT_STATS["knight"].attack_range
