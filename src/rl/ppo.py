"""PPO: rollout storage, GAE, and the clipped-surrogate update.

PPO rather than DQN because the action space here is factored and
conditional - a value-based method would want the flat 577 choices, which
throws away the sharing that makes the factored heads worth having.

Nothing exotic. The parts worth reading are:

- Masks are STORED with the rollout, not rebuilt at update time. The tile
  mask depends on the slot that was sampled, so rebuilding it would be a
  second implementation of a rule that already exists, and the two would
  drift.
- Bootstrapping respects the vector env's auto-reset: the value after a
  terminal step is zero, not the value of the fresh episode that replaced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .policy import ActorCritic


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    rollout_steps: int = 256      # per env, per update
    num_envs: int = 16
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.997          # ~1200-step episodes: credit has to travel
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatches: int = 4
    target_kl: float | None = 0.03
    seed: int = 0

    @property
    def batch_size(self) -> int:
        return self.rollout_steps * self.num_envs

    @property
    def minibatch_size(self) -> int:
        return max(1, self.batch_size // self.minibatches)

    @property
    def num_updates(self) -> int:
        return max(1, self.total_steps // self.batch_size)


@dataclass
class Rollout:
    """Fixed-size storage for one PPO iteration."""

    steps: int
    num_envs: int
    flat_size: int
    device: torch.device
    tile_count: int
    hand_size: int

    def __post_init__(self) -> None:
        s, n = self.steps, self.num_envs
        z = lambda *shape, dtype=torch.float32: torch.zeros(
            *shape, dtype=dtype, device=self.device
        )
        self.obs = z(s, n, self.flat_size)
        self.play = z(s, n, dtype=torch.long)
        self.slot = z(s, n, dtype=torch.long)
        self.tile = z(s, n, dtype=torch.long)
        self.logprob = z(s, n)
        self.value = z(s, n)
        self.reward = z(s, n)
        self.done = z(s, n)
        self.slot_mask = z(s, n, self.hand_size, dtype=torch.bool)
        self.tile_mask = z(s, n, self.tile_count, dtype=torch.bool)

    def flatten(self):
        return (
            self.obs.reshape(-1, self.flat_size),
            self.play.reshape(-1),
            self.slot.reshape(-1),
            self.tile.reshape(-1),
            self.logprob.reshape(-1),
            self.slot_mask.reshape(-1, self.hand_size),
            self.tile_mask.reshape(-1, self.tile_count),
        )


def compute_gae(
    rollout: Rollout,
    last_value: torch.Tensor,
    last_done: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over a rollout.

    ``done`` marks steps that ENDED an episode. The vector env auto-resets,
    so the state following a terminal step belongs to a different episode and
    must not be bootstrapped through - that is what ``nextnonterminal`` is
    for, and getting it wrong leaks reward across episode boundaries.
    """
    advantages = torch.zeros_like(rollout.reward)
    lastgaelam = torch.zeros_like(last_value)
    for t in reversed(range(rollout.steps)):
        if t == rollout.steps - 1:
            nextnonterminal = 1.0 - last_done
            nextvalue = last_value
        else:
            nextnonterminal = 1.0 - rollout.done[t + 1]
            nextvalue = rollout.value[t + 1]
        delta = (
            rollout.reward[t]
            + gamma * nextvalue * nextnonterminal
            - rollout.value[t]
        )
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + rollout.value


class PPO:
    """The optimiser side: holds the net, consumes rollouts, returns stats."""

    def __init__(self, net: ActorCritic, cfg: PPOConfig, device: torch.device):
        self.net = net
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    def set_lr(self, frac_remaining: float) -> float:
        lr = self.cfg.lr * frac_remaining if self.cfg.anneal_lr else self.cfg.lr
        for group in self.opt.param_groups:
            group["lr"] = lr
        return lr

    def update(
        self, rollout: Rollout, advantages: torch.Tensor, returns: torch.Tensor
    ) -> dict[str, float]:
        cfg = self.cfg
        obs, play, slot, tile, old_logprob, slot_mask, tile_mask = rollout.flatten()
        adv = advantages.reshape(-1)
        ret = returns.reshape(-1)
        old_value = rollout.value.reshape(-1)

        idx = np.arange(cfg.batch_size)
        clipfracs: list[float] = []
        stats: dict[str, float] = {}
        stop = False

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, device=self.device)

                newlogprob, entropy, newvalue = self.net.evaluate(
                    obs[mb_t], play[mb_t], slot[mb_t], tile[mb_t],
                    slot_mask[mb_t], tile_mask[mb_t],
                )
                logratio = newlogprob - old_logprob[mb_t]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                mb_adv = adv[mb_t]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg1, pg2).mean()

                # Clipped value loss, same spirit as the policy clip: stops a
                # single update moving the critic further than the data
                # supports.
                v_unclipped = (newvalue - ret[mb_t]) ** 2
                v_clipped = old_value[mb_t] + torch.clamp(
                    newvalue - old_value[mb_t], -cfg.clip_coef, cfg.clip_coef
                )
                v_loss = 0.5 * torch.max(
                    v_unclipped, (v_clipped - ret[mb_t]) ** 2
                ).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                stats = {
                    "policy_loss": pg_loss.item(),
                    "value_loss": v_loss.item(),
                    "entropy": ent_loss.item(),
                    "approx_kl": approx_kl,
                    "clipfrac": float(np.mean(clipfracs)),
                }

            if cfg.target_kl is not None and stats.get("approx_kl", 0) > cfg.target_kl:
                # The update has moved far enough; more epochs on this data
                # would be optimising against a stale ratio.
                stop = True
                break

        with torch.no_grad():
            y = ret.cpu().numpy()
            pred = old_value.cpu().numpy()
            var = np.var(y)
            stats["explained_variance"] = (
                float("nan") if var == 0 else float(1 - np.var(y - pred) / var)
            )
        stats["early_stop_epoch"] = float(epoch) if stop else float(cfg.update_epochs)
        return stats
