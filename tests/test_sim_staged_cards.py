"""Arena 2-4 cards, staged ahead of the vision model, and their mechanics.

These 14 cards are modelled but INERT: ``troop-counter/8`` cannot emit them,
so they have no observation channel and decks reject them. Adding them to the
manifest early is exactly the bug reverted in 309117a - every one becomes a
permanently-zero input. They go live when the model gains them.

The mechanics tested here (splash, kamikaze, charge, damage ramp, death
damage) are all read from the game data rather than invented, but the
BEHAVIOUR built on them is this project's, so it is worth pinning down.
"""

from __future__ import annotations

import pytest

from src.game.classes import ARENA_CLASSES, CARD_CLASSES
from src.sim import arena
from src.sim.engine import TICK_DT, Simulation, current_damage
from src.sim.units import STAGED_CLASSES, UNIT_STATS, Target

ARENA2 = {"skeleton", "valkyrie", "bomber", "tombstone"}
ARENA3 = {"barbarian", "battleram", "megaminion", "cannon"}
ARENA4 = {"wizard", "firespirit", "electrospirit", "skeletondragon",
          "infernotower", "bombtower"}


def spawn(sim, name, friendly, x, y):
    e = sim._spawn(name, friendly, x, y)
    e.deploy_remaining = 0.0
    return e


def run_for(sim, seconds):
    for _ in range(int(seconds / TICK_DT)):
        sim.tick()
        if sim.finished:
            return


class TestStaging:
    def test_all_three_arenas_are_present(self):
        assert ARENA2 | ARENA3 | ARENA4 <= set(UNIT_STATS)

    def test_staged_cards_are_marked_staged(self):
        assert STAGED_CLASSES == ARENA2 | ARENA3 | ARENA4

    def test_staged_cards_are_absent_from_the_detector_manifest(self):
        """The whole point: no dead observation channels."""
        assert not (STAGED_CLASSES & set(ARENA_CLASSES))
        assert not (STAGED_CLASSES & set(CARD_CLASSES))

    def test_live_cards_are_not_staged(self):
        assert not (set(ARENA_CLASSES) & STAGED_CLASSES)

    def test_a_deck_cannot_use_a_staged_card(self):
        import random

        from src.sim.env import Deck

        with pytest.raises(ValueError, match="detector cannot recognise"):
            Deck(("valkyrie", "knight", "giant", "archer", "minion"),
                 random.Random(0))

    def test_staged_units_can_still_be_simulated_directly(self):
        """Inert in decks, but the engine must handle them so the mechanics
        are exercised before the model bump rather than after."""
        sim = Simulation(seed=1)
        v = spawn(sim, "valkyrie", True, 4.5, 10.0)
        run_for(sim, 2.0)
        assert v.alive and v.y < 10.0


class TestSplash:
    def test_valkyrie_hits_several_units_at_once(self):
        sim = Simulation(seed=1)
        valk = spawn(sim, "valkyrie", True, 4.5, 9.0)
        victims = [spawn(sim, "skeleton", False, 4.5 + i * 0.15, 9.3)
                   for i in range(3)]
        valk.attack_cooldown = 0.0
        sim._retarget(valk)
        sim._tick_combat(valk, TICK_DT)
        assert sum(1 for v in victims if v.hp < v.max_hp) >= 2

    def test_single_target_units_hit_only_one(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        victims = [spawn(sim, "skeleton", False, 4.5 + i * 0.15, 9.3)
                   for i in range(3)]
        knight.attack_cooldown = 0.0
        sim._retarget(knight)
        sim._tick_combat(knight, TICK_DT)
        assert sum(1 for v in victims if v.hp < v.max_hp) == 1

    def test_splash_spares_friendly_units(self):
        sim = Simulation(seed=1)
        valk = spawn(sim, "valkyrie", True, 4.5, 9.0)
        friend = spawn(sim, "skeleton", True, 4.6, 9.2)
        spawn(sim, "skeleton", False, 4.5, 9.3)
        valk.attack_cooldown = 0.0
        sim._retarget(valk)
        sim._tick_combat(valk, TICK_DT)
        assert friend.hp == friend.max_hp

    @pytest.mark.parametrize("name", ["valkyrie", "bomber", "wizard",
                                      "skeletondragon", "bombtower"])
    def test_expected_units_have_splash(self, name):
        assert UNIT_STATS[name].splash_radius > 0.0

    @pytest.mark.parametrize("name", ["knight", "musketeer", "megaminion",
                                      "cannon"])
    def test_single_target_units_have_none(self, name):
        assert UNIT_STATS[name].splash_radius == 0.0


class TestKamikaze:
    @pytest.mark.parametrize("name", ["firespirit", "electrospirit", "battleram"])
    def test_marked_kamikaze(self, name):
        assert UNIT_STATS[name].kamikaze

    def test_a_spirit_dies_after_one_hit(self):
        sim = Simulation(seed=1)
        spirit = spawn(sim, "firespirit", True, 4.5, 9.0)
        victim = spawn(sim, "knight", False, 4.5, 9.2)
        spirit.attack_cooldown = 0.0
        sim._retarget(spirit)
        sim._tick_combat(spirit, TICK_DT)
        assert victim.hp < victim.max_hp
        assert not spirit.alive

    def test_a_battle_ram_becomes_two_barbarians(self):
        """Checked at the MOMENT the ram dies. Waiting lets the enemy tower
        shoot the barbarians and the test then measures tower dps instead."""
        sim = Simulation(seed=1)
        ram = spawn(sim, "battleram", True, 2.0, 4.2)
        assert UNIT_STATS["battleram"].spawn_on_death == 2
        assert UNIT_STATS["battleram"].spawns == "barbarian"

        for _ in range(int(20.0 / TICK_DT)):
            sim.tick()
            if not ram.alive:
                break
        assert not ram.alive, "ram never reached the tower"
        assert sum(1 for u in sim.units(True) if u.name == "barbarian") == 2

    def test_a_normal_unit_survives_attacking(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 9.0)
        spawn(sim, "skeleton", False, 4.5, 9.2)
        knight.attack_cooldown = 0.0
        sim._retarget(knight)
        sim._tick_combat(knight, TICK_DT)
        assert knight.alive


class TestCharge:
    def test_battle_ram_targets_buildings_only(self):
        assert UNIT_STATS["battleram"].targets is Target.BUILDINGS

    def test_charge_builds_over_distance_then_speeds_up(self):
        sim = Simulation(seed=1)
        ram = spawn(sim, "battleram", True, 2.0, 7.0)
        assert not ram.charging
        run_for(sim, 2.0)
        assert ram.charging, "never built a charge"

    def test_charging_moves_faster_than_walking(self):
        sim = Simulation(seed=1)
        slow = spawn(sim, "battleram", True, 2.0, 7.0)
        sim._retarget(slow)
        sim._move_toward(slow, sim.entities[slow.target_uid], TICK_DT)
        walked = abs(slow.y - 7.0)

        slow.charge_distance = UNIT_STATS["battleram"].charge_range * 10
        before = slow.y
        sim._move_toward(slow, sim.entities[slow.target_uid], TICK_DT)
        charged = abs(slow.y - before)
        assert charged > walked * 1.5

    def test_charge_is_spent_on_impact(self):
        sim = Simulation(seed=1)
        ram = spawn(sim, "battleram", True, 2.0, 4.0)
        ram.charge_distance = 99.0
        assert ram.charging
        sim._retarget(ram)
        ram.attack_cooldown = 0.0
        sim._tick_combat(ram, TICK_DT)
        assert ram.charge_distance == 0.0

    def test_non_charging_units_never_charge(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 10.0)
        run_for(sim, 5.0)
        assert not knight.charging
        assert UNIT_STATS["knight"].charge_range == 0.0


class TestDamageRamp:
    def test_inferno_tower_has_a_three_stage_ramp(self):
        ramp = UNIT_STATS["infernotower"].ramp
        assert len(ramp) == 3
        assert ramp[0][1] < ramp[1][1] < ramp[2][1]

    def test_damage_climbs_with_continuous_fire(self):
        sim = Simulation(seed=1)
        tower = spawn(sim, "infernotower", True, 4.5, 10.0)
        first = current_damage(tower)
        tower.fire_time = 10.0
        assert current_damage(tower) > first

    def test_the_ramp_resets_when_the_target_changes(self):
        """Switching targets is the counterplay to Inferno Tower."""
        sim = Simulation(seed=1)
        tower = spawn(sim, "infernotower", True, 4.5, 10.0)
        a = spawn(sim, "knight", False, 4.5, 10.5)
        sim._retarget(tower)
        tower.fire_time = 10.0
        assert current_damage(tower) > tower.stats.damage

        sim._damage(a, a.hp)
        spawn(sim, "knight", False, 4.6, 10.5)
        sim._retarget(tower)
        assert tower.fire_time == 0.0
        assert current_damage(tower) == pytest.approx(tower.stats.ramp[0][1])

    def test_units_without_a_ramp_use_flat_damage(self):
        sim = Simulation(seed=1)
        knight = spawn(sim, "knight", True, 4.5, 10.0)
        knight.fire_time = 30.0
        assert current_damage(knight) == knight.stats.damage


class TestDeathDamage:
    def test_bomb_tower_explodes_on_death(self):
        sim = Simulation(seed=1)
        tower = spawn(sim, "bombtower", True, 4.5, 10.0)
        victim = spawn(sim, "knight", False, 4.6, 10.2)
        assert UNIT_STATS["bombtower"].death_damage > 0
        sim._damage(tower, tower.hp)
        assert victim.hp < victim.max_hp

    def test_the_explosion_spares_friendly_units(self):
        sim = Simulation(seed=1)
        tower = spawn(sim, "bombtower", True, 4.5, 10.0)
        friend = spawn(sim, "knight", True, 4.6, 10.2)
        sim._damage(tower, tower.hp)
        assert friend.hp == friend.max_hp

    def test_the_explosion_has_a_finite_radius(self):
        sim = Simulation(seed=1)
        tower = spawn(sim, "bombtower", True, 4.5, 10.0)
        far = spawn(sim, "knight", False, 4.5, 14.0)
        sim._damage(tower, tower.hp)
        assert far.hp == far.max_hp


class TestSpawners:
    def test_tombstone_produces_skeletons_and_dies_into_more(self):
        sim = Simulation(seed=1)
        tomb = spawn(sim, "tombstone", True, 4.5, 11.0)
        run_for(sim, 8.0)
        assert any(u.name == "skeleton" for u in sim.units(True))

        assert UNIT_STATS["tombstone"].spawn_on_death == 4
        before = sum(1 for u in sim.units(True) if u.name == "skeleton")
        sim._damage(tomb, tomb.hp)
        after = sum(1 for u in sim.units(True) if u.name == "skeleton")
        assert after == before + 4

    def test_cannon_shoots_ground_only(self):
        assert UNIT_STATS["cannon"].targets is Target.GROUND
        assert UNIT_STATS["cannon"].damage > 0

    def test_defensive_buildings_expire(self):
        for name in ("cannon", "tombstone", "infernotower", "bombtower"):
            assert UNIT_STATS[name].lifetime > 0, name


class TestSquadSizes:
    @pytest.mark.parametrize("name,count", [
        ("skeleton", 3), ("barbarian", 5), ("skeletondragon", 2),
        ("archer", 2), ("minion", 3), ("goblin", 3), ("speargoblin", 3),
    ])
    def test_counts_match_the_real_cards(self, name, count):
        assert UNIT_STATS[name].count == count


class TestFlight:
    @pytest.mark.parametrize("name", ["minion", "megaminion", "skeletondragon"])
    def test_air_units_fly(self, name):
        assert UNIT_STATS[name].flying

    @pytest.mark.parametrize("name", ["knight", "valkyrie", "barbarian",
                                      "skeleton", "bomber", "wizard"])
    def test_ground_units_do_not(self, name):
        assert not UNIT_STATS[name].flying

    def test_ground_only_attackers_cannot_hit_air(self):
        sim = Simulation(seed=1)
        valk = spawn(sim, "valkyrie", True, 4.5, 9.0)
        drag = spawn(sim, "skeletondragon", False, 4.5, 9.1)
        from src.sim.engine import can_attack

        assert not can_attack(valk, drag)
