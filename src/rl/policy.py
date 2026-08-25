"""Actor-critic network with a MASKED, factored action space.

The action is three decisions, not one 577-way choice:

    play  in {no-op, play}
    slot  in 0..3        (only if play)
    tile  in 0..143      (only if play)

Factoring shares experience: every placement teaches the tile head about
locations and the slot head about card choice, instead of splitting samples
across 577 flat classes. It also keeps the network small enough that the
simulator, not the GPU, stays the bottleneck.

MASKING IS NOT OPTIONAL HERE. Without it the policy spends most of training
proposing illegal actions - unaffordable cards, tiles on the enemy half - and
learns to avoid them instead of learning to play. Three masks:

- ``play``: playing at all is illegal when no slot is affordable.
- ``slot``: ``hand_playable`` from the observation.
- ``tile``: DEPENDS ON THE SLOT. A spell may be aimed anywhere; a troop is
  confined to ``playable_mask``. So the slot is sampled FIRST and the tile
  mask built from it - an autoregressive factorisation, not an independent
  one. Masking tiles before the slot is known would forbid every legal spell
  target on the enemy half, which is the bug this project already fixed once
  in the placement rules.

Masked logits use a large FINITE negative rather than ``-inf``: an all-masked
row would otherwise produce NaN through the softmax and poison the update.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..env.actions import Action
from ..game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE

TILE_COUNT = ARENA_ROWS * ARENA_COLS
HIDDEN_SIZES = (512, 256)

# Finite stand-in for -inf. See module docstring.
NEG = -1e8


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Categorical over ``logits`` with ``mask`` (True = legal).

    A row with nothing legal falls back to uniform-over-everything rather
    than producing NaN. That should never happen - no-op is always legal and
    the friendly half is always placeable - so it is a guard, not a policy.
    """
    empty = ~mask.any(dim=-1, keepdim=True)
    safe = mask | empty
    return Categorical(logits=logits.masked_fill(~safe, NEG))


class RunningNorm(nn.Module):
    """Running mean/std over observations, frozen outside training.

    The observation mixes scales badly: almost everything is a 0/1 one-hot,
    but ``match_time`` runs to 300 and ``elixir`` to 10. Feeding that into a
    tanh trunk lets a handful of features dominate the first layer and
    saturate it. Normalising is cheaper and more general than special-casing
    the offending fields, and it keeps the observation SCHEMA untouched -
    that is a contract shared with the vision pipeline.

    State is saved with the checkpoint; a policy restored without its
    normaliser would be reading differently-scaled inputs.
    """

    def __init__(self, size: int, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bmean, bvar, bcount = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean += delta * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        self.var = (m_a + m_b + delta ** 2 * self.count * bcount / tot) / tot
        self.count += bcount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10, 10)


def _trunk(flat_size: int, h1: int, h2: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(flat_size, h1), nn.Tanh(),
        nn.Linear(h1, h2), nn.Tanh(),
    )


class ActorCritic(nn.Module):
    """SEPARATE actor and critic trunks, three policy heads, one value head.

    They were shared, and that was measurably wrong: the value loss produced
    a gradient 23x the policy's, almost all of it in the shared trunk, and
    ``max_grad_norm`` then scaled the whole thing down by 0.4. The critic
    learned beautifully (explained variance 0.91) while the actor did not
    move at all - KL 0.00001 over 146 updates. Separating them costs one
    extra trunk of compute, which is nothing next to the simulator.
    """

    def __init__(self, flat_size: int, hidden: tuple[int, int] = HIDDEN_SIZES):
        super().__init__()
        self.flat_size = flat_size
        self.hidden = tuple(hidden)
        h1, h2 = self.hidden
        self.norm = RunningNorm(flat_size)
        self.trunk = _trunk(flat_size, h1, h2)
        self.critic_trunk = _trunk(flat_size, h1, h2)
        self.play_head = nn.Linear(h2, 2)
        self.slot_head = nn.Linear(h2, HAND_SIZE)
        self.tile_head = nn.Linear(h2, TILE_COUNT)
        self.value_head = nn.Linear(h2, 1)
        self.apply(self._init)
        # Small final layers: near-uniform initial policy and a value head
        # that starts at roughly zero, which keeps early advantages honest.
        for head, gain in (
            (self.play_head, 0.01), (self.slot_head, 0.01),
            (self.tile_head, 0.01), (self.value_head, 1.0),
        ):
            nn.init.orthogonal_(head.weight, gain)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 2 ** 0.5)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        z = self.trunk(x)
        return (
            self.play_head(z),
            self.slot_head(z),
            self.tile_head(z),
            self.value_head(self.critic_trunk(x)).squeeze(-1),
        )

    def actor_parameters(self):
        for m in (self.trunk, self.play_head, self.slot_head, self.tile_head):
            yield from m.parameters()

    def critic_parameters(self):
        yield from self.critic_trunk.parameters()
        yield from self.value_head.parameters()

    # ----- masks -----

    @staticmethod
    def tile_mask_for(
        slot: torch.Tensor, hand_is_spell: torch.Tensor, playable: torch.Tensor
    ) -> torch.Tensor:
        """Legal tiles given the CHOSEN slot: everywhere for a spell, the
        troop mask otherwise."""
        is_spell = hand_is_spell.gather(1, slot.unsqueeze(1)) > 0.5
        return torch.where(is_spell, torch.ones_like(playable, dtype=torch.bool),
                           playable > 0.5)

    # ----- acting -----

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hand_playable: torch.Tensor,
        hand_is_spell: torch.Tensor,
        playable_mask: torch.Tensor,
        deterministic: bool = False,
    ):
        """Sample a batch of actions.

        Returns ``(play, slot, tile, logprob, value, slot_mask, tile_mask)``.
        The masks come back because PPO has to recompute these log-probs
        later against the SAME masks; rebuilding them at update time from a
        stored observation is a second implementation of the same rule and
        an easy place for the two to drift.
        """
        play_logits, slot_logits, tile_logits, value = self(obs)

        slot_mask = hand_playable > 0.5
        can_play = slot_mask.any(dim=-1)
        play_mask = torch.stack([torch.ones_like(can_play), can_play], dim=-1)

        play_dist = _masked_categorical(play_logits, play_mask)
        slot_dist = _masked_categorical(slot_logits, slot_mask)
        play = play_logits.argmax(-1) if deterministic else play_dist.sample()
        play = torch.where(can_play, play, torch.zeros_like(play))
        slot = slot_logits.masked_fill(~slot_mask, NEG).argmax(-1) if deterministic \
            else slot_dist.sample()

        tile_mask = self.tile_mask_for(slot, hand_is_spell, playable_mask)
        tile_dist = _masked_categorical(tile_logits, tile_mask)
        tile = tile_logits.masked_fill(~tile_mask, NEG).argmax(-1) if deterministic \
            else tile_dist.sample()

        acting = play == 1
        logprob = play_dist.log_prob(play) + acting * (
            slot_dist.log_prob(slot) + tile_dist.log_prob(tile)
        )
        return play, slot, tile, logprob, value, slot_mask, tile_mask

    def evaluate(
        self,
        obs: torch.Tensor,
        play: torch.Tensor,
        slot: torch.Tensor,
        tile: torch.Tensor,
        slot_mask: torch.Tensor,
        tile_mask: torch.Tensor,
    ):
        """Log-prob, entropy, value and per-head entropies of stored actions.

        The per-head dict is what makes a frozen placement head visible; see
        the comment on ``parts`` below.
        """
        play_logits, slot_logits, tile_logits, value = self(obs)

        can_play = slot_mask.any(dim=-1)
        play_mask = torch.stack([torch.ones_like(can_play), can_play], dim=-1)
        play_dist = _masked_categorical(play_logits, play_mask)
        slot_dist = _masked_categorical(slot_logits, slot_mask)
        tile_dist = _masked_categorical(tile_logits, tile_mask)

        acting = (play == 1).float()
        logprob = play_dist.log_prob(play) + acting * (
            slot_dist.log_prob(slot) + tile_dist.log_prob(tile)
        )
        # Component entropies summed UNWEIGHTED, which is a deliberate
        # exploration choice rather than the exact joint entropy.
        #
        # Weighting the slot/tile terms by P(play) is more correct, and it
        # was measurably wrong to do: the policy plays on ~4% of steps
        # (which is right - elixir only affords about that), so the
        # placement heads received 4% of the entropy bonus. For the tile
        # head that is an effective ent_coef of ~0.0004 against a 144-way
        # choice, i.e. no exploration pressure at all on the hardest
        # decision in the problem. Unweighted keeps those heads exploring
        # whether or not the policy is currently playing much.
        play_ent = play_dist.entropy()
        slot_ent = slot_dist.entropy()
        tile_ent = tile_dist.entropy()
        entropy = play_ent + slot_ent + tile_ent
        # Per-head, because the SUM hides which head is actually exploring.
        # Measured on run1/run2: play collapses to ~1% of its maximum while
        # slot and tile sit at ~96% of theirs, i.e. only one of the three
        # heads ever learned anything. That went unnoticed for two runs
        # because the sum alone looks like a healthy 5.7 of 7.05.
        parts = {"play": play_ent, "slot": slot_ent, "tile": tile_ent}
        return logprob, entropy, value, parts


def to_action(play: int, slot: int, tile: int) -> Action:
    """Network output -> the env's ``Action``."""
    if play == 0:
        return Action.no_op()
    return Action(hand_index=int(slot), tile_x=int(tile) % ARENA_COLS,
                  tile_y=int(tile) // ARENA_COLS)
