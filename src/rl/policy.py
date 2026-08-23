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


class ActorCritic(nn.Module):
    """Shared trunk, three policy heads and a value head."""

    def __init__(self, flat_size: int, hidden: tuple[int, int] = HIDDEN_SIZES):
        super().__init__()
        self.flat_size = flat_size
        self.hidden = tuple(hidden)
        h1, h2 = self.hidden
        self.trunk = nn.Sequential(
            nn.Linear(flat_size, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )
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
        z = self.trunk(x)
        return (
            self.play_head(z),
            self.slot_head(z),
            self.tile_head(z),
            self.value_head(z).squeeze(-1),
        )

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
        """Log-prob, entropy and value of stored actions, for the PPO update."""
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
        entropy = (
            play_dist.entropy() + slot_dist.entropy() + tile_dist.entropy()
        )
        return logprob, entropy, value


def to_action(play: int, slot: int, tile: int) -> Action:
    """Network output -> the env's ``Action``."""
    if play == 0:
        return Action.no_op()
    return Action(hand_index=int(slot), tile_x=int(tile) % ARENA_COLS,
                  tile_y=int(tile) // ARENA_COLS)
