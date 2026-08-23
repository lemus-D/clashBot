"""Tests for the detector class manifest and the naming contract.

These cover the two bugs this module exists to prevent: a class list that
disagrees with the detector, and a class ORDER that changes underneath
existing recordings and checkpoints.
"""

from __future__ import annotations

import pytest

from src.game.classes import (
    ARENA_CLASSES,
    ARENA_INDEX,
    CARD_CLASSES,
    CARD_INDEX,
    IGNORED_ARENA_CLASSES,
    derive_from_class_names,
    merge_preserving_order,
    normalize_name,
)

# A miniature stand-in for get_model(...).class_names, with the same shape as
# troop-counter/8: matching blue/red sets, a spawn-only unit with no card,
# and the two tower classes.
FAKE_CLASS_NAMES = [
    "blue knight", "red knight", "card knight",
    "blue mini pekka", "red mini pekka", "card mini pekka",
    "blue goblin brawler", "red goblin brawler",  # spawn-only: no card
    "blue king tower", "red king tower",
    "blue princess tower", "red princess tower",
]


class TestDerive:
    def test_splits_arena_and_card_classes(self):
        arena, cards = derive_from_class_names(FAKE_CLASS_NAMES)
        assert arena == ["goblinbrawler", "knight", "minipekka"]
        assert cards == ["knight", "minipekka"]

    def test_spawn_only_unit_is_arena_but_not_card(self):
        """The Goblin Brawler category: in the arena, never in hand.

        This is why there are two index spaces. A shared list would give the
        hand a one-hot channel that can never fire.
        """
        arena, cards = derive_from_class_names(FAKE_CLASS_NAMES)
        assert "goblinbrawler" in arena
        assert "goblinbrawler" not in cards

    def test_towers_are_excluded_from_arena(self):
        arena, _ = derive_from_class_names(FAKE_CLASS_NAMES)
        assert not (set(arena) & IGNORED_ARENA_CLASSES)

    def test_unrecognised_prefix_raises(self):
        """A new prefix means the parser no longer understands the model.

        Skipping it would put a silent hole in the observation, which is
        exactly the failure mode this module exists to stop.
        """
        with pytest.raises(ValueError, match="prefixes"):
            derive_from_class_names(FAKE_CLASS_NAMES + ["green wizard"])

    def test_asymmetric_sides_raise(self):
        """The arena tensor packs both sides over ONE class list."""
        with pytest.raises(ValueError, match="blue-only"):
            derive_from_class_names(FAKE_CLASS_NAMES + ["blue wizard"])

    def test_normalization_strips_separators_and_case(self):
        arena, _ = derive_from_class_names(
            ["blue Spear-Goblin", "red spear_goblin"]
        )
        assert arena == ["speargoblin"]


class TestAppendOnlyOrder:
    """Class order is channel order. Reordering silently reinterprets every
    recorded observation and makes a trained checkpoint unmigratable."""

    def test_new_names_are_appended_not_sorted_in(self):
        merged = merge_preserving_order(["knight", "minion"], ["archer", "knight"])
        assert merged == ["knight", "minion", "archer"]

    def test_existing_indices_never_move(self):
        existing = ["knight", "minion", "giant"]
        merged = merge_preserving_order(existing, ["archer", "bomber", "giant"])
        for i, name in enumerate(existing):
            assert merged.index(name) == i

    def test_vanished_names_are_kept_as_dead_channels(self):
        """Dropping a removed class would shift every later index."""
        merged = merge_preserving_order(["knight", "minion"], ["knight"])
        assert merged == ["knight", "minion"]

    def test_merge_is_idempotent(self):
        once = merge_preserving_order(["knight"], ["archer", "knight"])
        twice = merge_preserving_order(once, ["archer", "knight"])
        assert once == twice


class TestLoadedManifest:
    """The real manifest, as the rest of the code sees it."""

    def test_indices_agree_with_lists(self):
        assert ARENA_INDEX == {n: i for i, n in enumerate(ARENA_CLASSES)}
        assert CARD_INDEX == {n: i for i, n in enumerate(CARD_CLASSES)}

    def test_names_are_normalized(self):
        for name in (*ARENA_CLASSES, *CARD_CLASSES):
            assert name == normalize_name(name)

    def test_no_duplicates(self):
        assert len(set(ARENA_CLASSES)) == len(ARENA_CLASSES)
        assert len(set(CARD_CLASSES)) == len(CARD_CLASSES)

    def test_towers_absent(self):
        assert not (set(ARENA_CLASSES) & IGNORED_ARENA_CLASSES)

    def test_every_card_is_also_an_arena_unit(self):
        """A card you can play must be visible once played.

        The reverse does NOT hold - that is the spawn-only category.
        """
        assert set(CARD_CLASSES) <= set(ARENA_CLASSES)

    def test_every_card_has_an_elixir_cost(self):
        """cards.py validates this at import; assert it rather than trust it,
        since a missing cost makes hand_playable silently wrong."""
        from src.game.cards import CARD_COSTS

        assert set(CARD_COSTS) == set(CARD_CLASSES)
        assert all(1 <= c <= 10 for c in CARD_COSTS.values())
