"""Tests for observation encoding and the schema contract.

The arena tensor is the large majority of the observation and is invisible
when it goes wrong - a troop the encoder cannot place looks identical to an
empty tile. These tests assert that the encoding actually happens.
"""

from __future__ import annotations

import numpy as np

from src.env.observation import (
    OBSERVATION_SHAPES,
    ObservationBuilder,
    schema_descriptor,
    schema_hash,
)
from src.game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE, GameBoard
from src.game.cards import Card, Troop
from src.game.classes import ARENA_CLASSES, ARENA_INDEX, CARD_CLASSES, CARD_INDEX
from src.game.state import GameState


def make_board() -> GameBoard:
    return GameBoard(monitor_width=1000, monitor_height=1600)


class TestArenaEncoding:
    def test_friendly_and_enemy_use_separate_channel_halves(self):
        board = make_board()
        board.troops_in_arena[9][2] = Troop("knight", "blue", 2, 9)
        board.troops_in_arena[3][4] = Troop("knight", "red", 4, 3)
        tensor = board.to_tensor()

        n = len(ARENA_CLASSES)
        idx = ARENA_INDEX["knight"]
        assert tensor[9, 2, idx] == 1.0
        assert tensor[3, 4, n + idx] == 1.0
        assert tensor.sum() == 2.0

    def test_spawn_only_unit_encodes_in_arena(self):
        """Goblin Brawler has no card but must still be seen on the field."""
        board = make_board()
        board.troops_in_arena[5][5] = Troop("goblin brawler", "red", 5, 5)
        tensor = board.to_tensor()
        assert tensor.sum() == 1.0

    def test_detector_style_names_encode(self):
        """The model emits SINGULAR, space-separated names. Plurals here once
        blanked four of the most common units out of every observation."""
        board = make_board()
        for i, name in enumerate(["minion", "archer", "spear goblin", "goblin"]):
            board.troops_in_arena[i][0] = Troop(name, "blue", 0, i)
        assert board.to_tensor().sum() == 4.0

    def test_unknown_troop_is_dropped_but_warns(self, caplog):
        board = make_board()
        board.troops_in_arena[1][1] = Troop("wizard", "blue", 1, 1)
        with caplog.at_level("WARNING"):
            tensor = board.to_tensor()
        assert tensor.sum() == 0.0
        assert "ARENA_CLASSES" in caplog.text

    def test_towers_are_dropped_silently(self):
        """Excluding them is a decision, not a bug, so no warning."""
        board = make_board()
        board.troops_in_arena[0][0] = Troop("princess tower", "red", 0, 0)
        assert board.to_tensor().sum() == 0.0


class TestHandEncoding:
    def test_hand_is_keyed_on_card_classes(self):
        board = make_board()
        board.cards_in_hand = [Card("minion"), None, Card("giant"), None]
        hand = board.hand_to_tensor()

        assert hand.shape == (HAND_SIZE, len(CARD_CLASSES))
        assert hand[0, CARD_INDEX["minion"]] == 1.0
        assert hand[2, CARD_INDEX["giant"]] == 1.0
        assert hand.sum() == 2.0

    def test_hand_has_no_channel_for_spawn_only_units(self):
        assert "goblinbrawler" in ARENA_INDEX
        assert "goblinbrawler" not in CARD_INDEX

    def test_costs_resolve_from_the_card_table(self):
        board = make_board()
        board.cards_in_hand = [Card("giant"), Card("goblin"), None, None]
        costs = board.hand_costs()
        assert list(costs) == [5.0, 2.0, 0.0, 0.0]


class TestSchema:
    def test_flatten_matches_declared_shapes(self):
        obs = ObservationBuilder().build(make_board(), GameState())
        flat = ObservationBuilder.flatten(obs)
        expected = sum(int(np.prod(s)) for s in OBSERVATION_SHAPES.values())
        assert flat.shape == (expected,)
        assert flat.dtype == np.float32

    def test_every_declared_field_is_built(self):
        obs = ObservationBuilder().build(make_board(), GameState())
        assert set(obs) == set(OBSERVATION_SHAPES)
        for key, shape in OBSERVATION_SHAPES.items():
            assert np.asarray(obs[key]).shape == shape, key

    def test_hand_and_arena_are_sized_by_different_lists(self):
        assert OBSERVATION_SHAPES["hand"] == (HAND_SIZE, len(CARD_CLASSES))
        assert OBSERVATION_SHAPES["arena"] == (
            ARENA_ROWS, ARENA_COLS, len(ARENA_CLASSES) * 2,
        )

    def test_descriptor_carries_both_class_lists(self):
        """A recording has to say what its channels MEAN, or it cannot be
        re-encoded under a later schema."""
        d = schema_descriptor()
        assert d["arena_classes"] == list(ARENA_CLASSES)
        assert d["card_classes"] == list(CARD_CLASSES)
        assert d["model_id"]

    def test_hash_is_stable(self):
        assert schema_hash() == schema_hash()

    def test_hash_changes_when_a_class_is_added(self, monkeypatch):
        """Adding a class must invalidate old recordings and checkpoints.

        The hash is the only thing standing between a new class list and
        silently reinterpreting every previously recorded observation, since
        adding an arena class shifts the whole enemy channel half.
        """
        import src.env.observation as obs_mod

        before = schema_hash()
        monkeypatch.setattr(
            obs_mod, "ARENA_CLASSES", obs_mod.ARENA_CLASSES + ("wizard",)
        )
        assert schema_hash() != before
