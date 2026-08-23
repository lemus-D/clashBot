"""Tests for the PPO training stack.

Most of these are about MASKING. An RL agent will exploit any illegal action
the environment happens to tolerate, and a mask bug shows up as a policy that
trains fine and then behaves nonsensically - so the invariants are asserted
directly rather than inferred from a learning curve.

The GAE tests exist because the vector env auto-resets: a bootstrap that
leaks across an episode boundary is silent, plausible, and wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.actions import Action
from src.env.observation import ObservationBuilder, schema_hash
from src.game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE
from src.rl.evaluate import BENCHMARK, evaluate, format_results, headline
from src.rl.policy import NEG, TILE_COUNT, ActorCritic, to_action
from src.rl.ppo import PPO, PPOConfig, Rollout, compute_gae
from src.rl.vec_env import VecSimEnv

DEVICE = torch.device("cpu")
FLAT = ObservationBuilder().flat_size()


def net() -> ActorCritic:
    torch.manual_seed(0)
    return ActorCritic(FLAT).to(DEVICE)


def batch(n=8):
    torch.manual_seed(1)
    return torch.randn(n, FLAT)


class TestActionMapping:
    def test_no_op_ignores_slot_and_tile(self):
        a = to_action(0, 3, 100)
        assert a.is_no_op

    def test_tile_index_round_trips(self):
        for tile in (0, 1, ARENA_COLS - 1, ARENA_COLS, TILE_COUNT - 1):
            a = to_action(1, 2, tile)
            assert a.tile_y * ARENA_COLS + a.tile_x == tile
            assert 0 <= a.tile_x < ARENA_COLS and 0 <= a.tile_y < ARENA_ROWS

    def test_slot_is_carried_through(self):
        assert to_action(1, 3, 5).hand_index == 3


class TestMasking:
    def test_unaffordable_slots_are_never_sampled(self):
        m = net()
        n = 64
        playable = torch.zeros(n, HAND_SIZE)
        playable[:, 2] = 1.0  # only slot 2 affordable
        _, slot, _, _, _, _, _ = m.act(
            batch(n), playable, torch.zeros(n, HAND_SIZE), torch.ones(n, TILE_COUNT)
        )
        assert (slot == 2).all()

    def test_playing_is_impossible_with_nothing_affordable(self):
        m = net()
        n = 64
        play, *_ = m.act(
            batch(n), torch.zeros(n, HAND_SIZE), torch.zeros(n, HAND_SIZE),
            torch.ones(n, TILE_COUNT),
        )
        assert (play == 0).all(), "policy tried to play with no affordable card"

    def test_troops_are_confined_to_the_playable_mask(self):
        m = net()
        n = 64
        playable_tiles = torch.zeros(n, TILE_COUNT)
        playable_tiles[:, 100:110] = 1.0
        _, _, tile, _, _, _, tile_mask = m.act(
            batch(n), torch.ones(n, HAND_SIZE), torch.zeros(n, HAND_SIZE),
            playable_tiles,
        )
        assert ((tile >= 100) & (tile < 110)).all()
        assert tile_mask[:, :100].sum() == 0

    def test_a_spell_may_be_aimed_anywhere(self):
        """The whole reason the tile mask is conditioned on the slot."""
        m = net()
        n = 64
        playable_tiles = torch.zeros(n, TILE_COUNT)
        playable_tiles[:, 100:110] = 1.0
        hand_is_spell = torch.ones(n, HAND_SIZE)
        _, _, tile, _, _, _, tile_mask = m.act(
            batch(n), torch.ones(n, HAND_SIZE), hand_is_spell, playable_tiles,
        )
        assert tile_mask.all(), "spell was confined to the troop mask"
        assert (tile < 100).any(), "never sampled outside the troop mask"

    def test_the_tile_mask_follows_the_chosen_slot(self):
        m = net()
        hand_is_spell = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        playable = torch.zeros(1, TILE_COUNT)
        playable[0, :5] = 1.0

        troop = m.tile_mask_for(torch.tensor([0]), hand_is_spell, playable)
        spell = m.tile_mask_for(torch.tensor([1]), hand_is_spell, playable)
        assert troop.sum() == 5
        assert spell.all()

    def test_an_all_masked_row_does_not_produce_nan(self):
        """Should be unreachable, but a NaN here poisons the whole update."""
        m = net()
        n = 4
        _, _, _, logprob, value, _, _ = m.act(
            batch(n), torch.ones(n, HAND_SIZE), torch.zeros(n, HAND_SIZE),
            torch.zeros(n, TILE_COUNT),  # nothing placeable at all
        )
        assert torch.isfinite(logprob).all()
        assert torch.isfinite(value).all()

    def test_deterministic_mode_is_repeatable(self):
        m = net()
        args = (batch(8), torch.ones(8, HAND_SIZE), torch.zeros(8, HAND_SIZE),
                torch.ones(8, TILE_COUNT))
        a = m.act(*args, deterministic=True)
        b = m.act(*args, deterministic=True)
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])


class TestLogProbConsistency:
    def test_act_and_evaluate_agree(self):
        """PPO's ratio is meaningless if collection and update disagree."""
        m = net()
        n = 32
        obs = batch(n)
        playable = torch.ones(n, HAND_SIZE)
        spells = torch.zeros(n, HAND_SIZE)
        tiles = torch.ones(n, TILE_COUNT)

        play, slot, tile, logprob, _, slot_mask, tile_mask = m.act(
            obs, playable, spells, tiles
        )
        again, entropy, value = m.evaluate(
            obs, play, slot, tile, slot_mask, tile_mask
        )
        assert torch.allclose(logprob, again, atol=1e-5)
        assert torch.isfinite(entropy).all() and (entropy >= 0).all()
        assert value.shape == (n,)

    def test_a_no_op_logprob_ignores_slot_and_tile(self):
        m = net()
        n = 16
        obs = batch(n)
        slot_mask = torch.ones(n, HAND_SIZE, dtype=torch.bool)
        tile_mask = torch.ones(n, TILE_COUNT, dtype=torch.bool)
        play = torch.zeros(n, dtype=torch.long)

        a, _, _ = m.evaluate(obs, play, torch.zeros(n, dtype=torch.long),
                             torch.zeros(n, dtype=torch.long), slot_mask, tile_mask)
        b, _, _ = m.evaluate(obs, play, torch.full((n,), 3, dtype=torch.long),
                             torch.full((n,), 99, dtype=torch.long),
                             slot_mask, tile_mask)
        assert torch.allclose(a, b)

    def test_entropy_is_not_suppressed_by_a_low_play_rate(self):
        """Weighting slot/tile entropy by P(play) gave the placement heads
        ~4% of the exploration bonus, which is none at all against a 144-way
        choice. Unweighted keeps them exploring."""
        m = net()
        n = 16
        _, entropy, _ = m.evaluate(
            batch(n), torch.zeros(n, dtype=torch.long),
            torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long),
            torch.ones(n, HAND_SIZE, dtype=torch.bool),
            torch.ones(n, TILE_COUNT, dtype=torch.bool),
        )
        # ln(144) alone is ~4.97; a suppressed formulation lands near zero.
        assert entropy.mean() > 4.0


class TestGAE:
    def _rollout(self, steps=4, envs=1):
        r = Rollout(steps, envs, 2, DEVICE, TILE_COUNT, HAND_SIZE)
        return r

    def test_no_reward_no_advantage(self):
        r = self._rollout()
        adv, ret = compute_gae(r, torch.zeros(1), torch.zeros(1), 0.99, 0.95)
        assert torch.allclose(adv, torch.zeros_like(adv))
        assert torch.allclose(ret, torch.zeros_like(ret))

    def test_a_terminal_step_does_not_bootstrap(self):
        """The env auto-resets, so the state after a done belongs to a
        different episode. Bootstrapping through it leaks reward."""
        r = self._rollout(steps=2)
        r.reward[0, 0] = 1.0
        r.value[1, 0] = 100.0    # next episode looks great
        r.done[1, 0] = 1.0       # ...but step 0 ended the last one

        adv, _ = compute_gae(r, torch.zeros(1), torch.zeros(1), 0.99, 0.95)
        assert adv[0, 0] == pytest.approx(1.0), "leaked value across a reset"

    def test_reward_propagates_backwards_without_a_terminal(self):
        """Credit reaches earlier steps DISCOUNTED, so advantage decreases
        the further back you go from the reward."""
        r = self._rollout(steps=3)
        r.reward[2, 0] = 1.0
        adv, _ = compute_gae(r, torch.zeros(1), torch.zeros(1), 0.99, 1.0)
        assert adv[2, 0] > adv[1, 0] > adv[0, 0] > 0, "credit did not travel back"
        assert adv[0, 0] == pytest.approx(0.99 ** 2, rel=1e-4)
        assert adv[1, 0] == pytest.approx(0.99, rel=1e-4)

    def test_returns_are_advantages_plus_values(self):
        r = self._rollout(steps=3)
        r.reward.normal_()
        r.value.normal_()
        adv, ret = compute_gae(r, torch.zeros(1), torch.zeros(1), 0.99, 0.95)
        assert torch.allclose(ret, adv + r.value, atol=1e-5)


class TestPPOUpdate:
    def test_an_update_runs_and_changes_the_policy(self):
        m = net()
        cfg = PPOConfig(rollout_steps=8, num_envs=4, minibatches=2,
                        update_epochs=2, target_kl=None)
        algo = PPO(m, cfg, DEVICE)
        r = Rollout(cfg.rollout_steps, cfg.num_envs, FLAT, DEVICE,
                    TILE_COUNT, HAND_SIZE)
        r.obs.normal_()
        r.reward.normal_()
        r.slot_mask[:] = True
        r.tile_mask[:] = True
        r.logprob.normal_()

        before = m.play_head.weight.detach().clone()
        adv, ret = compute_gae(r, torch.zeros(cfg.num_envs),
                               torch.zeros(cfg.num_envs), 0.99, 0.95)
        stats = algo.update(r, adv, ret)

        assert not torch.allclose(before, m.play_head.weight)
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            assert np.isfinite(stats[key]), key

    def test_lr_annealing_reaches_zero(self):
        m = net()
        cfg = PPOConfig(lr=1e-3, anneal_lr=True)
        algo = PPO(m, cfg, DEVICE)
        assert algo.set_lr(1.0) == pytest.approx(1e-3)
        assert algo.set_lr(0.0) == pytest.approx(0.0)


class TestVecEnv:
    def test_shapes_line_up_with_the_policy(self):
        envs = VecSimEnv(num_envs=4, seed=1)
        obs, masks = envs.observe()
        assert obs.shape == (4, envs.flat_size)
        assert masks.hand_playable.shape == (4, HAND_SIZE)
        assert masks.hand_is_spell.shape == (4, HAND_SIZE)
        assert masks.playable.shape == (4, TILE_COUNT)
        envs.close()

    def test_each_env_gets_its_own_opponent_instance(self):
        """A shared bot would have one committed lane driven by four matches."""
        envs = VecSimEnv(num_envs=4, seed=1, opponents=("tankandsupport",))
        ids = {id(e.opponent) for e in envs.envs}
        assert len(ids) == 4
        envs.close()

    def test_opponents_are_assigned_round_robin(self):
        envs = VecSimEnv(num_envs=4, seed=1, opponents=("cycler", "control"))
        assert envs._opponent_names == ["cycler", "control", "cycler", "control"]
        envs.close()

    def test_auto_reset_reports_the_finished_episode(self):
        envs = VecSimEnv(num_envs=2, seed=1, opponents=("idle",))
        seen = None
        for _ in range(1400):
            _, dones, infos = envs.step([Action.no_op()] * 2)
            done_info = [i for i in infos if "episode" in i]
            if done_info:
                seen = done_info[0]["episode"]
                break
        assert seen is not None, "no episode finished"
        assert seen["length"] > 0
        assert seen["result"] in ("win", "loss", "draw")
        assert seen["opponent"] == "idle"
        envs.close()

    def test_the_batch_keeps_producing_after_an_episode_ends(self):
        envs = VecSimEnv(num_envs=2, seed=1, opponents=("idle",))
        for _ in range(1400):
            envs.step([Action.no_op()] * 2)
        obs, _ = envs.observe()
        assert np.isfinite(obs).all()
        envs.close()


class TestEvaluate:
    def test_it_scores_every_opponent(self):
        m = net()
        results = evaluate(m, DEVICE, ("idle", "cycler"), episodes=2, batch=2)
        assert [r.opponent for r in results] == ["idle", "cycler"]
        for r in results:
            assert r.episodes == 2
            assert r.wins + r.losses + r.draws == 2

    def test_unknown_opponent_raises(self):
        with pytest.raises(ValueError, match="Unknown opponents"):
            evaluate(net(), DEVICE, ("nope",), episodes=1)

    def test_headline_excludes_idle(self):
        """Every working policy beats idle, so including it inflates the
        number without carrying signal."""
        results = evaluate(net(), DEVICE, ("idle", "cycler"), episodes=2, batch=2)
        cycler = next(r for r in results if r.opponent == "cycler")
        # Only cycler contributes, so the headline IS its win rate.
        assert headline(results) == pytest.approx(cycler.wins / cycler.episodes)

        idle_only = [r for r in results if r.opponent == "idle"]
        assert headline(idle_only) == 0.0, "idle alone should carry no signal"
        assert "OVERALL (ex-idle)" in format_results(results)

    def test_the_benchmark_covers_every_registered_opponent(self):
        from src.sim.opponents import OPPONENTS

        assert set(BENCHMARK) == set(OPPONENTS)


class TestCheckpoint:
    def test_a_mismatched_schema_is_refused(self, tmp_path):
        """The observation schema has changed four times; loading across a
        change would read the wrong channels while appearing to work."""
        import torch as _t

        from src.rl.train import load_checkpoint

        path = tmp_path / "bad.pt"
        _t.save({"model_state": net().state_dict(), "flat_size": FLAT,
                 "hidden": [512, 256], "schema_hash": "deadbeef"}, path)
        with pytest.raises(ValueError, match="schema"):
            load_checkpoint(str(path), DEVICE)

    def test_a_matching_checkpoint_round_trips(self, tmp_path):
        from src.rl.ppo import PPOConfig
        from src.rl.train import load_checkpoint, save_checkpoint

        path = str(tmp_path / "ok.pt")
        original = net()
        save_checkpoint(path, original, PPOConfig(), 1234, {})
        loaded, ckpt = load_checkpoint(path, DEVICE)

        assert ckpt["schema_hash"] == schema_hash()
        assert ckpt["global_step"] == 1234
        for a, b in zip(original.parameters(), loaded.parameters()):
            assert torch.allclose(a, b)
