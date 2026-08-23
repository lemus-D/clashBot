"""The control archetype, the style table, and per-episode deck sampling.

Control is the first scripted bot that converts a won defence into an attack.
That matters for what a policy learns: beating BigSpender teaches you to
punish over-commitment, but beating Control requires not over-committing
yourself, because whatever survives your push comes straight back.
"""

from __future__ import annotations

import random

import pytest

from src.env.actions import Action
from src.game.cards import SPELL_CARDS
from src.game.classes import CARD_CLASSES
from src.sim.env import DEFAULT_DECK, Deck, DeckSpread, ObservationNoise, SimEnv
from src.sim.opponents import (
    LANES,
    OPPONENTS,
    PUSH_ROW,
    STYLES,
    BigSpender,
    Control,
    Cycler,
    OpponentView,
    ScriptedOpponent,
    TankAndSupport,
    UnitView,
    make_opponent,
)


def view(hand, elixir, threats=(), own=(), time=0.0) -> OpponentView:
    return OpponentView(
        hand=list(hand), elixir=elixir, time=time, phase="normal",
        can_place=lambda tx, ty: True, threats=tuple(threats),
        own_units=tuple(own),
    )


def unit(tile_x=4, tile_y=10, name="knight") -> UnitView:
    return UnitView(name=name, tile_x=tile_x, tile_y=tile_y, flying=False)


class TestStyleTable:
    @pytest.mark.parametrize("cls", [BigSpender, Cycler, TankAndSupport, Control])
    def test_every_bot_names_a_real_style(self, cls):
        assert cls.style_name in STYLES

    @pytest.mark.parametrize("name", sorted(STYLES))
    def test_styles_are_sane(self, name):
        st = STYLES[name]
        assert 0.3 <= st.reaction_s <= 2.0
        assert 0.0 <= st.reserve <= 8.0
        assert st.tank_row >= st.push_row

    def test_reaction_delay_comes_from_the_style(self):
        assert Cycler(seed=1).reaction_s == STYLES["cycle"].reaction_s
        assert Control(seed=1).reaction_s == STYLES["control"].reaction_s

    def test_beatdown_starts_its_tank_deeper_than_cycle_chips(self):
        assert STYLES["beatdown"].tank_row > STYLES["cycle"].push_row


class TestControl:
    def test_counter_pushes_behind_a_surviving_defender(self):
        opp = Control(seed=1)
        v = view(["knight", "musketeer"], 10.0, own=[unit(tile_x=6, tile_y=9)])
        move = opp(v)
        assert move is not None
        _, tx, ty = move
        assert tx == 6, "support went to the wrong lane"
        assert ty == 10, "support should land BEHIND the survivor"

    def test_it_supports_the_most_advanced_survivor(self):
        opp = Control(seed=1)
        v = view(["knight"], 10.0,
                 own=[unit(tile_x=1, tile_y=13), unit(tile_x=7, tile_y=9)])
        _, tx, _ = opp(v)
        assert tx == 7

    def test_it_keeps_a_defensive_reserve_when_counter_pushing(self):
        """Spending down to nothing on a counter-push means the next attack
        goes unanswered, which is precisely what control play avoids."""
        opp = Control(seed=1)
        reserve = STYLES["control"].reserve
        v = view(["giant"], reserve + 1.0, own=[unit()])
        assert opp(v) is None, "spent into the reserve"

        v = view(["giant"], reserve + 6.0, own=[unit()])
        assert opp(v) is not None

    def test_it_holds_elixir_with_nothing_to_support(self):
        opp = Control(seed=1)
        assert opp(view(["knight", "goblin"], 6.0)) is None

    def test_it_chips_rather_than_overflow(self):
        opp = Control(seed=1)
        move = opp(view(["knight", "goblin"], 10.0))
        assert move is not None
        _, lane, row = move
        assert lane in LANES and row == PUSH_ROW

    def test_defence_still_takes_priority_over_a_counter_push(self):
        opp = Control(seed=1)
        v = view(["goblin", "knight"], 10.0,
                 threats=[unit(tile_x=2, tile_y=13)], own=[unit(tile_x=7)])
        opp(v)                       # first sighting starts the clock
        v.time = opp.reaction_s + 0.1
        _, tx, _ = opp(v)
        assert tx == 2, "counter-pushed while being attacked"

    def test_own_units_on_the_enemy_half_are_not_survivors(self):
        """Something already through is attacking, not waiting for support."""
        v = view(["knight"], 10.0, own=[unit(tile_y=3)])
        assert v.survivors() == []

    def test_it_plays_a_legal_full_match(self):
        env = SimEnv(seed=5, randomize_scale=0.0, noise=ObservationNoise.off(),
                     opponent=Control(seed=5))
        env.reset()
        done = False
        deployed = 0
        while not done:
            before = len(env.sim.units(False))
            _, _, done, info = env.step(Action.no_op())
            deployed += max(0, len(env.sim.units(False)) - before)
        assert deployed > 0, "control never played anything"
        assert info["result"] in ("win", "loss", "draw")

    def test_control_is_registered(self):
        assert "control" in OPPONENTS
        assert isinstance(make_opponent("control", seed=1), Control)


class TestDeckSpread:
    def test_samples_only_detectable_cards(self):
        spread = DeckSpread()
        rng = random.Random(0)
        for _ in range(50):
            for card in spread.sample(rng):
                assert card in CARD_CLASSES

    def test_samples_the_requested_size_without_duplicates(self):
        spread = DeckSpread(size=8)
        deck = spread.sample(random.Random(0))
        assert len(deck) == 8 and len(set(deck)) == 8

    def test_enough_troops_to_be_playable(self):
        spread = DeckSpread(min_troops=5)
        rng = random.Random(0)
        for _ in range(50):
            deck = spread.sample(rng)
            assert sum(1 for c in deck if c not in SPELL_CARDS) >= 5

    def test_decks_actually_vary(self):
        spread = DeckSpread()
        rng = random.Random(0)
        seen = {tuple(sorted(spread.sample(rng))) for _ in range(40)}
        assert len(seen) > 1

    def test_off_returns_the_fixed_deck(self):
        assert DeckSpread.off().sample(random.Random(0)) == DEFAULT_DECK

    def test_a_sampled_deck_is_always_constructible(self):
        spread = DeckSpread()
        rng = random.Random(0)
        for _ in range(30):
            Deck(spread.sample(rng), random.Random(0))


class TestEnvDeckSampling:
    def test_decks_change_between_episodes(self):
        env = SimEnv(seed=5, noise=ObservationNoise.off())
        seen = set()
        for _ in range(25):
            env.reset()
            seen.add(tuple(sorted(env._deck._order)))
        assert len(seen) > 1

    def test_the_two_sides_draw_independently(self):
        env = SimEnv(seed=5, noise=ObservationNoise.off())
        differed = False
        for _ in range(25):
            env.reset()
            if set(env._deck._order) != set(env._opp_deck._order):
                differed = True
                break
        assert differed, "both sides always got the same deck"

    def test_off_pins_both_sides_to_the_default(self):
        env = SimEnv(seed=5, decks=DeckSpread.off(),
                     noise=ObservationNoise.off())
        for _ in range(5):
            env.reset()
            assert set(env._deck._order) == set(DEFAULT_DECK)

    def test_the_hand_is_still_encoded_whatever_the_deck(self):
        """Deck variety is only survivable because hand identity is IN the
        observation - the policy can see which cards it holds."""
        from src.game.board import HAND_SIZE

        env = SimEnv(seed=5, noise=ObservationNoise.off())
        for _ in range(15):
            obs = env.reset()
            assert obs["hand"].sum() == HAND_SIZE

    def test_sampled_decks_never_contain_staged_cards(self):
        from src.sim.units import STAGED_CLASSES

        env = SimEnv(seed=5, noise=ObservationNoise.off())
        for _ in range(25):
            env.reset()
            assert not (set(env._deck._order) & STAGED_CLASSES)
