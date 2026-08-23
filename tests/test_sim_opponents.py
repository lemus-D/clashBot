"""Tests for the scripted opponents.

These are the fixed yardstick the policy gets measured against, so a bug in
one of them silently corrupts every benchmark taken with it. The deadlock
test in particular is here because the bug was real: TankAndSupport with no
expensive card in hand played nothing, and playing nothing meant the hand
never cycled, so it sat at capped elixir for the whole match.
"""

from __future__ import annotations

import pytest

from src.env.actions import Action
from src.game.classes import CARD_CLASSES
from src.sim.env import ObservationNoise, SimEnv
from src.sim.opponents import (
    DEFEND_ROW,
    LANES,
    OPPONENTS,
    PUSH_ROW,
    SUPPORT_ROW,
    BigSpender,
    Cycler,
    Idle,
    OpponentView,
    TankAndSupport,
    make_opponent,
)
from src.sim.units import SPELL_NAMES


def view(hand, elixir, can_place=lambda tx, ty: True) -> OpponentView:
    return OpponentView(
        hand=list(hand), elixir=elixir, time=0.0,
        phase="normal", can_place=can_place,
    )


def play_match(opponent, seed=5, limit=1400):
    """Run a full match against a do-nothing policy; return the final info."""
    env = SimEnv(seed=seed, randomize_scale=0.0,
                 noise=ObservationNoise.off(), opponent=opponent)
    env.reset()
    done = False
    info = {}
    deployed = 0
    while not done and limit:
        before = len(env.sim.units(False))
        _, _, done, info = env.step(Action.no_op())
        after = len(env.sim.units(False))
        deployed += max(0, after - before)
        limit -= 1
    return info, deployed


class TestView:
    def test_affordable_is_cheapest_first(self):
        v = view(["giant", "goblin", "knight"], 10.0)
        assert [v.hand[i] for i in v.affordable()] == ["goblin", "knight", "giant"]

    def test_affordable_respects_elixir(self):
        v = view(["giant", "goblin"], 2.0)
        assert [v.hand[i] for i in v.affordable()] == ["goblin"]

    def test_spells_are_excluded_by_default(self):
        v = view(["fireball", "knight"], 10.0)
        assert [v.hand[i] for i in v.affordable()] == ["knight"]
        assert "fireball" in SPELL_NAMES


class TestIdle:
    def test_plays_nothing(self):
        assert Idle()(view(["giant"], 10.0)) is None

    def test_a_match_against_idle_is_a_draw(self):
        info, deployed = play_match(Idle())
        assert deployed == 0
        assert info["result"] == "draw"


class TestBigSpender:
    def test_holds_below_the_spend_threshold(self):
        assert BigSpender(seed=1)(view(["goblin"], 3.0)) is None

    def test_plays_the_most_expensive_affordable_card(self):
        v = view(["goblin", "giant", "knight"], 10.0)
        slot, _, _ = BigSpender(seed=1)(v)
        assert v.hand[slot] == "giant"

    def test_falls_back_to_what_it_can_pay_for(self):
        # spend_above=0 isolates the card CHOICE from the hold threshold:
        # at 4 elixir the giant is out of reach, so it takes the next best.
        v = view(["goblin", "giant", "knight"], 4.0)
        slot, _, _ = BigSpender(seed=1, spend_above=0.0)(v)
        assert v.hand[slot] == "knight"

    def test_places_at_the_bridge_in_a_lane(self):
        _, lane, row = BigSpender(seed=1)(view(["giant"], 10.0))
        assert lane in LANES and row == PUSH_ROW

    def test_actually_pressures_in_a_real_match(self):
        info, deployed = play_match(BigSpender(seed=5))
        assert deployed > 0
        assert info["crowns"][1] > 0, "BigSpender never took a tower"


class TestCycler:
    def test_plays_the_cheapest_affordable_card(self):
        v = view(["giant", "goblin", "knight"], 10.0)
        slot, _, _ = Cycler(seed=1)(v)
        assert v.hand[slot] == "goblin"

    def test_plays_whenever_it_can_afford_anything(self):
        assert Cycler(seed=1)(view(["goblin"], 2.0)) is not None

    def test_holds_when_nothing_is_affordable(self):
        assert Cycler(seed=1)(view(["giant"], 1.0)) is None

    def test_deploys_more_often_than_bigspender(self):
        _, cycler_units = play_match(Cycler(seed=5))
        _, spender_units = play_match(BigSpender(seed=5))
        assert cycler_units > spender_units


class TestTankAndSupport:
    def test_waits_for_an_expensive_card(self):
        assert TankAndSupport(seed=1)(view(["goblin", "knight"], 6.0)) is None

    def test_commits_a_tank_when_it_can_follow_up(self):
        v = view(["giant", "goblin"], 8.0)
        slot, lane, row = TankAndSupport(seed=1)(v)
        assert v.hand[slot] == "giant"
        assert row == PUSH_ROW and lane in LANES

    def test_holds_the_tank_without_follow_up_elixir(self):
        """A tank walking in alone dies to the first thing it meets."""
        assert TankAndSupport(seed=1)(view(["giant", "goblin"], 5.0)) is None

    def test_support_goes_behind_the_tank_in_the_same_lane(self):
        opp = TankAndSupport(seed=1)
        _, lane, _ = opp(view(["giant", "goblin"], 8.0))
        slot, support_lane, row = opp(view(["giant", "goblin"], 3.0))
        assert support_lane == lane
        assert row == SUPPORT_ROW
        assert SUPPORT_ROW > PUSH_ROW, "support must be BEHIND the tank"

    def test_the_push_is_held_until_support_lands(self):
        """Failing to afford support must not abandon the push - otherwise
        this bot degenerates into BigSpender."""
        opp = TankAndSupport(seed=1)
        _, lane, _ = opp(view(["giant", "goblin"], 8.0))
        assert opp(view(["giant", "goblin"], 0.0)) is None
        slot, support_lane, row = opp(view(["giant", "goblin"], 4.0))
        assert support_lane == lane and row == SUPPORT_ROW

    def test_cycles_out_of_a_hand_with_no_tank(self):
        """The deadlock case: all-cheap hand, capped elixir. It must play
        SOMETHING or the hand never rotates toward a tank."""
        opp = TankAndSupport(seed=1)
        v = view(["goblin", "goblin", "knight", "minion"], 10.0)
        move = opp(v)
        assert move is not None, "deadlocked on a hand with no tank"
        assert move[2] == DEFEND_ROW, "cycling should go at the back, not the bridge"

    def test_does_not_cycle_at_low_elixir(self):
        opp = TankAndSupport(seed=1)
        assert opp(view(["goblin", "knight"], 5.0)) is None

    def test_no_hand_can_stall_it_for_a_whole_match(self):
        """Property check over every 4-card hand the deck can produce."""
        import itertools

        opp = TankAndSupport(seed=1)
        for hand in itertools.combinations(CARD_CLASSES, 4):
            opp._lane = None
            assert opp(view(hand, 10.0)) is not None, f"stalled on {hand}"

    def test_builds_pushes_in_a_real_match(self):
        info, deployed = play_match(TankAndSupport(seed=5))
        assert deployed > 0


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(OPPONENTS))
    def test_every_registered_opponent_plays_a_legal_match(self, name):
        info, _ = play_match(make_opponent(name, seed=5))
        assert info["result"] in ("win", "loss", "draw")

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown opponent"):
            make_opponent("nope")

    @pytest.mark.parametrize("name", sorted(OPPONENTS))
    def test_opponents_are_deterministic(self, name):
        a = play_match(make_opponent(name, seed=5))
        b = play_match(make_opponent(name, seed=5))
        assert a == b

    def test_opponents_cannot_see_the_policy_hand_or_elixir(self):
        """A benchmark that can cheat is not a benchmark."""
        env = SimEnv(seed=5, randomize_scale=0.0, opponent=Idle())
        v = env._opponent_view()
        assert not hasattr(v, "sim")
        assert v.hand == list(env._opp_deck.hand)
        assert v.hand != env.hand or v.elixir == env.sim.elixir[False]
