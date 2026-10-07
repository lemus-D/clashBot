"""Spells may be cast anywhere; troops may not.

Before this rule existed, ``playable_mask`` was the only placement gate and
it confined everything to the friendly half. Fireball and Arrows could
therefore never reach an enemy tower, and a policy would have correctly
learned that two of its twelve cards were worthless - a false lesson that
would have transferred straight to real hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.env.actions import Action
from src.env.observation import OBSERVATION_SHAPES, ObservationBuilder
from src.game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE, GameBoard
from src.game.cards import SPELL_CARDS, Card, is_spell
from src.game.classes import CARD_CLASSES
from src.sim.env import ObservationNoise, SimEnv
from src.sim.units import SPELL_DAMAGE

ENEMY_TILE = (4, 2)     # deep on the enemy half
FRIENDLY_TILE = (4, 12)


def board_with(hand) -> GameBoard:
    b = GameBoard(ARENA_COLS * 64, ARENA_ROWS * 64)
    for i, name in enumerate(hand):
        b.cards_in_hand[i] = Card(name) if name else None
    return b


class TestCardData:
    def test_spells_are_real_cards(self):
        assert SPELL_CARDS <= set(CARD_CLASSES)

    def test_is_spell_normalizes(self):
        for variant in ("fireball", "Fireball", "Fire Ball", "fire-ball"):
            assert is_spell(variant), variant
        assert not is_spell("knight")

    def test_spell_damage_table_matches_the_card_table(self):
        assert set(SPELL_DAMAGE) == SPELL_CARDS


class TestPlacementRule:
    def test_troops_cannot_be_placed_on_the_enemy_half(self):
        b = board_with(["knight"])
        assert not b.is_placeable(*ENEMY_TILE)

    def test_spells_can_be_placed_on_the_enemy_half(self):
        b = board_with(["fireball"])
        assert b.is_placeable(*ENEMY_TILE, spell=True)

    def test_spells_can_be_placed_on_the_bridge_row(self):
        """Row 7 is barred for ground troops but is a legal spell target."""
        b = board_with(["fireball"])
        assert not b.is_placeable(4, 7)
        assert b.is_placeable(4, 7, spell=True)

    def test_spells_still_respect_the_arena_bounds(self):
        b = board_with(["fireball"])
        assert not b.is_placeable(-1, 4, spell=True)
        assert not b.is_placeable(ARENA_COLS, 4, spell=True)

    def test_spell_mask_covers_the_whole_arena(self):
        b = board_with(["fireball"])
        assert b.get_placeable_mask(spell=True).sum() == ARENA_ROWS * ARENA_COLS

    def test_troop_mask_is_unchanged(self):
        b = board_with(["knight"])
        mask = b.get_placeable_mask()
        assert mask[:7].sum() == 0     # enemy half
        assert mask[7].sum() == 0      # bridge row
        assert mask[8:].sum() == 8 * ARENA_COLS


class TestObservation:
    def test_hand_is_spell_flags_the_right_slots(self):
        b = board_with(["knight", "fireball", None, "arrows"])
        flags = b.hand_is_spell()
        assert list(flags) == [0.0, 1.0, 0.0, 1.0]

    def test_empty_slots_are_not_spells(self):
        assert board_with([None] * HAND_SIZE).hand_is_spell().sum() == 0

    def test_schema_carries_the_flags(self):
        assert OBSERVATION_SHAPES["hand_is_spell"] == (HAND_SIZE,)

    def test_observation_includes_the_flags(self):
        env = SimEnv(seed=11, randomize_scale=0.0, noise=ObservationNoise.off())
        obs = env.reset()
        assert obs["hand_is_spell"].shape == (HAND_SIZE,)
        expected = [1.0 if is_spell(c) else 0.0 for c in env.hand]
        assert list(obs["hand_is_spell"]) == expected


class TestSimulator:
    @staticmethod
    def _env_with_spell_in_hand(seed_range=range(40)):
        """Find a seed whose opening hand holds a spell."""
        for seed in seed_range:
            env = SimEnv(seed=seed, randomize_scale=0.0,
                         noise=ObservationNoise.off())
            env.reset()
            for slot, name in enumerate(env.hand):
                if is_spell(name):
                    return env, slot, name
        pytest.skip("no seed produced a spell in the opening hand")

    def test_a_spell_can_be_cast_on_the_enemy_half(self):
        env, slot, _ = self._env_with_spell_in_hand()
        env.sim.elixir[True] = 10.0
        _, _, _, info = env.step(Action(slot, *ENEMY_TILE))
        assert info["action_ok"], info["action_reason"]

    def test_a_spell_cast_on_an_enemy_tower_damages_it(self):
        """The whole point of the fix."""
        env, slot, _ = self._env_with_spell_in_hand()
        env.sim.elixir[True] = 10.0
        tower = next(t for t in env.sim.towers(False) if t.tower_key == "enemy_left")
        before = tower.hp
        env.step(Action(slot, int(tower.x), int(tower.y)))
        assert tower.hp < before

    def test_a_troop_still_cannot_go_on_the_enemy_half(self):
        env = SimEnv(seed=11, randomize_scale=0.0, noise=ObservationNoise.off())
        env.reset()
        troop_slot = next(
            i for i, n in enumerate(env.hand) if not is_spell(n)
        )
        env.sim.elixir[True] = 10.0
        _, _, _, info = env.step(Action(troop_slot, *ENEMY_TILE))
        assert not info["action_ok"]
        assert info["action_reason"] == "tile_not_placeable"

    def test_casting_a_spell_spends_elixir_and_cycles_the_card(self):
        env, slot, name = self._env_with_spell_in_hand()
        env.sim.elixir[True] = 10.0
        env.step(Action(slot, *ENEMY_TILE))
        assert env.sim.elixir[True] < 10.0
        assert env.hand[slot] != name

    def test_a_spell_leaves_no_unit_behind(self):
        env, slot, _ = self._env_with_spell_in_hand()
        env.sim.elixir[True] = 10.0
        before = len(env.sim.units(True))
        env.step(Action(slot, *ENEMY_TILE))
        assert len(env.sim.units(True)) == before


class TestPolicyMasking:
    def test_slot_must_be_resolved_before_the_tile_mask(self):
        """A tile mask chosen before the slot forbids every legal spell
        target on the enemy half. This asserts the two masks actually
        differ, which is what makes the ordering matter."""
        env = SimEnv(seed=11, randomize_scale=0.0, noise=ObservationNoise.off())
        obs = env.reset()
        troop_mask = np.asarray(obs["playable_mask"]).reshape(-1) > 0
        spell_mask = np.ones(ARENA_ROWS * ARENA_COLS, dtype=bool)
        assert troop_mask.sum() < spell_mask.sum()

    def test_random_policy_can_target_the_enemy_half_with_a_spell(self):
        from src.sim.run import RandomSimPolicy

        # Seed chosen for a spell in the opening hand - a hand with no spell
        # would pass this test vacuously by never having the chance to fail.
        env, slot, _ = TestSimulator._env_with_spell_in_hand()
        env.sim.elixir[True] = 10.0
        obs = env.observe()
        assert obs["hand_is_spell"][slot] > 0
        pol = RandomSimPolicy(no_op_prob=0.0, seed=1)

        saw_enemy_half_spell = False
        for _ in range(400):
            a = pol(obs)
            if a.is_no_op:
                continue
            if obs["hand_is_spell"][a.hand_index] > 0 and a.tile_y < 7:
                saw_enemy_half_spell = True
                break
        assert saw_enemy_half_spell, "policy never aimed a spell at the enemy half"
