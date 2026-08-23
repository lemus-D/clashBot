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
    BACKLINE_ROW,
    DEFEND_ROW,
    LANES,
    OPPONENTS,
    PUSH_ROW,
    SUPPORT_ROW,
    Threat,
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
        """Deep, not at the bridge: beatdown wants the push to gather."""
        v = view(["giant", "goblin"], 8.0)
        slot, lane, row = TankAndSupport(seed=1)(v)
        assert v.hand[slot] == "giant"
        assert row == BACKLINE_ROW and lane in LANES
        assert BACKLINE_ROW > PUSH_ROW, "tank should start behind the bridge"

    def test_holds_the_tank_without_follow_up_elixir(self):
        """A tank walking in alone dies to the first thing it meets."""
        assert TankAndSupport(seed=1)(view(["giant", "goblin"], 5.0)) is None

    def test_support_goes_behind_the_tank_in_the_same_lane(self):
        opp = TankAndSupport(seed=1)
        _, lane, _ = opp(view(["giant", "goblin"], 8.0))
        slot, support_lane, row = opp(view(["giant", "goblin"], 3.0))
        assert support_lane == lane
        assert row > BACKLINE_ROW, "support must be BEHIND the tank"

    def test_the_push_is_held_until_support_lands(self):
        """Failing to afford support must not abandon the push - otherwise
        this bot degenerates into BigSpender."""
        opp = TankAndSupport(seed=1)
        _, lane, _ = opp(view(["giant", "goblin"], 8.0))
        assert opp(view(["giant", "goblin"], 0.0)) is None
        slot, support_lane, row = opp(view(["giant", "goblin"], 4.0))
        assert support_lane == lane and row > BACKLINE_ROW

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


def threat(tile_x=4, tile_y=12, name="knight", flying=False) -> Threat:
    return Threat(name=name, tile_x=tile_x, tile_y=tile_y, flying=flying)


def after_reaction(opp, v):
    """Drive a bot past its reaction delay and return the resulting move.

    The clock starts on FIRST SIGHTING, so one call at a large timestamp
    never defends - the first call is the sighting. Two calls are needed,
    which is also how it plays out tick by tick in a real match.
    """
    opp(v)
    v.time += opp.reaction_s + 0.05
    return opp(v)


class TestDefence:
    """None of these bots defended at all before, which is why a random
    policy beat them by walking cards into an empty lane."""

    def test_the_defensive_response_waits_for_the_reaction_delay(self):
        opp = Cycler(seed=1)
        v = view(["goblin", "knight"], 10.0)
        v.threats = (threat(),)
        v.time = 0.0
        assert opp._defend(v) is None, "reacted on the frame it appeared"

        v.time = opp.reaction_s + 0.05
        assert opp._defend(v) is not None

    def test_it_keeps_attacking_while_it_has_not_reacted_yet(self):
        """A human mid-push does not freeze the instant a threat lands. The
        delay gates the DEFENSIVE response, not the whole bot - and spending
        elixir on offence in that window is a realistic, punishable mistake."""
        opp = Cycler(seed=1)
        v = view(["goblin", "knight"], 10.0)
        v.threats = (threat(),)
        v.time = 0.0
        move = opp(v)
        assert move is not None and move[2] == PUSH_ROW

    def test_defence_answers_the_threats_lane(self):
        opp = Cycler(seed=1)
        v = view(["goblin", "knight"], 10.0)
        v.threats = (threat(tile_x=7),)
        _, tx, _ = after_reaction(opp, v)
        assert tx == 7

    def test_defence_is_placed_in_the_invaders_path(self):
        opp = Cycler(seed=1)
        v = view(["goblin"], 10.0)
        v.threats = (threat(tile_y=11),)
        _, _, row = after_reaction(opp, v)
        assert row == 12, "should intercept between the threat and the king"

    def test_the_deepest_invader_is_answered_first(self):
        """The deepest one is about to hit a tower; the near one is not."""
        opp = Cycler(seed=1)
        v = view(["goblin"], 10.0)
        v.threats = (threat(tile_x=1, tile_y=9), threat(tile_x=7, tile_y=13))
        _, tx, _ = after_reaction(opp, v)
        assert tx == 7

    def test_enemies_on_their_own_half_are_not_invaders(self):
        v = view(["goblin"], 10.0)
        v.threats = (threat(tile_y=3),)
        assert v.invaders() == []

    def test_building_targeters_are_not_used_to_defend(self):
        """A Giant walks past whatever is attacking you."""
        v = view(["giant"], 10.0)
        assert v.defenders() == []
        assert v.affordable() == [0], "giant should still be affordable"

    def test_defence_takes_priority_over_attacking(self):
        opp = BigSpender(seed=1)
        v = view(["goblin", "giant"], 10.0)
        v.threats = (threat(tile_y=13),)
        _, _, row = after_reaction(opp, v)
        assert row > PUSH_ROW, "pushed instead of defending"

    def test_the_delay_resets_once_the_half_is_clear(self):
        opp = Cycler(seed=1)
        v = view(["goblin"], 10.0)
        v.threats = (threat(),)
        v.time = 0.0
        opp(v)

        v.threats = ()
        v.time = 5.0
        opp(v)

        v.threats = (threat(),)
        v.time = 5.05
        assert opp._defend(v) is None, "timer did not reset when the half cleared"

    def test_a_sustained_push_is_answered_once_not_re_delayed(self):
        """Reinforcements arriving must not restart the clock, or a steady
        stream of units would keep the bot permanently unable to respond."""
        opp = Cycler(seed=1)
        v = view(["goblin"], 10.0)
        v.threats = (threat(tile_x=1),)
        v.time = 0.0
        opp(v)
        v.threats = (threat(tile_x=1), threat(tile_x=2))
        v.time = opp.reaction_s + 0.05
        assert opp._defend(v) is not None

    @pytest.mark.parametrize("cls", [BigSpender, Cycler, TankAndSupport])
    def test_every_bot_defends(self, cls):
        opp = cls(seed=1)
        v = view(["goblin", "knight", "giant"], 10.0)
        v.threats = (threat(tile_y=13),)
        move = after_reaction(opp, v)
        assert move is not None and move[2] > PUSH_ROW

    @pytest.mark.parametrize("cls", [BigSpender, Cycler, TankAndSupport])
    def test_reaction_delays_are_human_scale(self, cls):
        assert 0.3 <= cls.reaction_s <= 2.0


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
