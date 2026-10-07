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
from ..env.observation import FIELD_OFFSETS, field_indices_excluding
from ..game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE, TOWER_TILES
from ..game.classes import ARENA_CLASSES
from ..game.state import TOWER_KEYS

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


class FactoredPolicy(nn.Module):
    """Masking, sampling and log-probs for the factored action space.

    Architecture-agnostic on purpose: subclasses provide ``forward`` returning
    ``(play_logits, slot_logits, tile_logits, value)`` and inherit all of the
    masking below. The autoregressive slot-then-tile ordering is subtle enough
    (see the module docstring) that a second copy of it would drift, which is
    the mistake this project already made with the placement predicate.
    """

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def actor_parameters(self):
        raise NotImplementedError

    def critic_parameters(self):
        raise NotImplementedError

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



class ActorCritic(FactoredPolicy):
    """FLAT MLP baseline. Separate actor and critic trunks, three policy
    heads, one value head.

    Kept selectable alongside ``ConvActorCritic`` so the two can be compared
    on the same seed. Its weakness is structural and measured: ``tile_head``
    is ``Linear(h2, 144)``, i.e. 144 independent weight vectors with nothing
    connecting tile 37 to tile 38, so a lesson learned at one tile teaches
    its neighbour nothing. Tile entropy sat at ~97% of the maximum the
    placement mask allows after 2M steps in both run1 and run2.

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


def to_action(play: int, slot: int, tile: int) -> Action:
    """Network output -> the env's ``Action``."""
    if play == 0:
        return Action.no_op()
    return Action(hand_index=int(slot), tile_x=int(tile) % ARENA_COLS,
                  tile_y=int(tile) // ARENA_COLS)


# --------------------------------------------------------------------------
# Conv architecture
# --------------------------------------------------------------------------

CONV_WIDTH = 32          # feature channels in the conv stack
VEC_WIDTH = 32           # channels the non-spatial vector is broadcast into


class _SpatialInput(nn.Module):
    """Rebuild the flat observation's spatial structure as conv channels.

    ``flatten()`` destroys the arena's shape on the way in, which is why a
    flat MLP has to learn 144 unrelated tile logits: nothing tells it that
    tile 37 neighbours 38. Everything here is either already in the
    observation or a constant, so the observation SCHEMA is untouched - the
    contract shared with the vision pipeline does not move, no recording is
    invalidated, and ``schema_hash`` stays put.

    Channels, ``2 * |ARENA_CLASSES| + 7``:

    - ``2 * |ARENA_CLASSES|``  arena one-hots, friendly then enemy (reshaped)
    - 1   ``playable_mask``, already a 16x9 map and previously flattened. By
          construction it carries the river, the friendly half, and any lane
          opened by a destroyed tower - the arena's fixed geometry, free.
    - 2   normalised row and column index. A conv is translation
          EQUIVARIANT, and Clash Royale placement emphatically is not: row 15
          is your king's pocket, row 8 is the bridge. Without these the
          shared kernel could not tell them apart and would be strictly
          worse than the MLP it replaces. Not a refinement - load-bearing.
    - 2   tower footprint, friendly then enemy. Constant 1 on a tower's tile.
          Needed because a destroyed tower reads HP 0 and so does every
          empty tile, so HP alone cannot say "tower dead here" as opposed to
          "no tower here" - a distinction that decides late-match play.
    - 2   tower HP, friendly then enemy, painted at the tower's own tile.
          ``tower_hp`` is otherwise six scalars, forcing the network to learn
          an arbitrary binding from "index 0" to "the left lane". Painted, a
          damaged tower is a LOCAL feature and "defend the hurt side" is one
          pattern the shared kernel learns once for both sides.

    The towers are not in ``ARENA_CLASSES`` at all - tower skins make them
    detect unreliably, so they sit in ``IGNORED_ARENA_CLASSES`` - which means
    nothing in the arena one-hots ever marks a tower's location. These
    channels are the only spatial evidence of where the towers are.
    """

    def __init__(self) -> None:
        super().__init__()
        arena_off, arena_size = FIELD_OFFSETS["arena"]
        mask_off, mask_size = FIELD_OFFSETS["playable_mask"]
        hp_off, hp_size = FIELD_OFFSETS["tower_hp"]
        self.arena_slice = (arena_off, arena_off + arena_size)
        self.mask_slice = (mask_off, mask_off + mask_size)
        self.hp_slice = (hp_off, hp_off + hp_size)
        self.arena_channels = len(ARENA_CLASSES) * 2

        rows = torch.arange(ARENA_ROWS, dtype=torch.float32) / (ARENA_ROWS - 1)
        cols = torch.arange(ARENA_COLS, dtype=torch.float32) / (ARENA_COLS - 1)
        coords = torch.stack([
            rows.view(-1, 1).expand(ARENA_ROWS, ARENA_COLS),
            cols.view(1, -1).expand(ARENA_ROWS, ARENA_COLS),
        ])
        self.register_buffer("coords", coords)

        # Tower tiles come from ``board.TOWER_TILES`` in ``TOWER_KEYS`` order,
        # which is the order ``tower_hp`` is built in, so index i of the HP
        # vector belongs to tile i here.
        friendly = [i for i, k in enumerate(TOWER_KEYS) if k.startswith("friendly")]
        enemy = [i for i, k in enumerate(TOWER_KEYS) if k.startswith("enemy")]
        if len(friendly) + len(enemy) != len(TOWER_KEYS):
            raise ValueError(
                f"Every tower key must be friendly_* or enemy_* so it can be "
                f"assigned a side; got {TOWER_KEYS}."
            )

        def flat_tiles(idxs: list[int]) -> torch.Tensor:
            out = []
            for i in idxs:
                row, col = TOWER_TILES[TOWER_KEYS[i]]
                out.append(row * ARENA_COLS + col)
            return torch.tensor(out, dtype=torch.long)

        self.register_buffer("friendly_hp_idx", torch.tensor(friendly, dtype=torch.long))
        self.register_buffer("enemy_hp_idx", torch.tensor(enemy, dtype=torch.long))
        self.register_buffer("friendly_tiles", flat_tiles(friendly))
        self.register_buffer("enemy_tiles", flat_tiles(enemy))

        footprint = torch.zeros(2, ARENA_ROWS * ARENA_COLS)
        footprint[0, flat_tiles(friendly)] = 1.0
        footprint[1, flat_tiles(enemy)] = 1.0
        self.register_buffer("footprint", footprint.view(2, ARENA_ROWS, ARENA_COLS))

    @property
    def channels(self) -> int:
        return self.arena_channels + 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        tiles = ARENA_ROWS * ARENA_COLS

        # (N, ROWS, COLS, C) -> (N, C, ROWS, COLS). flatten() writes the arena
        # channels-last, so this permute inverts that rather than guessing.
        arena = x[:, self.arena_slice[0]:self.arena_slice[1]]
        arena = arena.view(n, ARENA_ROWS, ARENA_COLS, self.arena_channels)
        arena = arena.permute(0, 3, 1, 2)

        mask = x[:, self.mask_slice[0]:self.mask_slice[1]].view(
            n, 1, ARENA_ROWS, ARENA_COLS
        )

        hp = x[:, self.hp_slice[0]:self.hp_slice[1]]
        painted = x.new_zeros(n, 2, tiles)
        painted[:, 0].scatter_(
            1, self.friendly_tiles.expand(n, -1),
            hp.index_select(1, self.friendly_hp_idx),
        )
        painted[:, 1].scatter_(
            1, self.enemy_tiles.expand(n, -1),
            hp.index_select(1, self.enemy_hp_idx),
        )
        painted = painted.view(n, 2, ARENA_ROWS, ARENA_COLS)

        return torch.cat([
            arena,
            mask,
            self.coords.expand(n, -1, -1, -1),
            self.footprint.expand(n, -1, -1, -1),
            painted,
        ], dim=1)


class _VectorNorm(RunningNorm):
    """``RunningNorm`` over only the NON-spatial fields of the observation.

    The spatial channels are all naturally 0..1 - one-hots, a boolean mask,
    normalised coordinates, and HP fill fractions - so they need no
    rescaling, and normalising a sparse one-hot by its own tiny variance
    would amplify noise instead. The vector half genuinely does mix scales
    (``match_time`` to 300, ``elixir`` to 10), so it keeps the normaliser.

    ``update`` and ``forward`` both take the FULL flat observation and slice
    internally, so ``train.py``'s ``net.norm.update(obs)`` needs no change.
    """

    def __init__(self, index: torch.Tensor):
        super().__init__(int(index.numel()))
        self.register_buffer("index", index)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        RunningNorm.update(self, x.index_select(-1, self.index))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RunningNorm.forward(self, x.index_select(-1, self.index))


class _ConvTrunk(nn.Module):
    """Conv stack over the arena, with the non-spatial vector broadcast in.

    Two 3x3 layers before the vector joins, then one after: a tile's features
    then depend on a 5x5 neighbourhood of the arena AND on elixir, hand and
    tower state. Broadcasting is what lets "I hold a Fireball" or "my left
    tower is at 30%" interact with POSITION at all; without it the conv could
    only reason about the board in isolation.
    """

    def __init__(self, in_channels: int, vec_size: int,
                 width: int = CONV_WIDTH, vec_width: int = VEC_WIDTH):
        super().__init__()
        self.board = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.vec = nn.Sequential(nn.Linear(vec_size, vec_width), nn.ReLU())
        self.mix = nn.Sequential(
            nn.Conv2d(width + vec_width, width, 3, padding=1), nn.ReLU(),
        )
        self.width = width

    def forward(self, spatial: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        z = self.board(spatial)
        v = self.vec(vec)[:, :, None, None].expand(-1, -1, *z.shape[-2:])
        return self.mix(torch.cat([z, v], dim=1))


class ConvActorCritic(FactoredPolicy):
    """Conv over the 9x16 arena, so placement GENERALISES across tiles.

    The flat baseline learns 144 independent tile logits and needs examples at
    every tile separately; measured, its tile head never left ~97% of the
    entropy its mask allows, across two full runs and 2M steps. Here the tile
    head is a 1x1 conv over the board, so all 144 logits come from ONE shared
    kernel and every placement trains it.

    That trades absolute position for weight sharing, which for this game
    would be a bad trade on its own - see ``_SpatialInput`` for the coordinate
    channels that buy it back, and the learned per-tile bias below, which
    restores the MLP's one real strength: memorising that a tile is special.

    Actor and critic keep SEPARATE trunks, as in ``ActorCritic`` - sharing one
    let the value loss dominate the gradient and the actor never moved.
    """

    def __init__(self, flat_size: int, hidden: tuple[int, int] = HIDDEN_SIZES,
                 width: int = CONV_WIDTH):
        super().__init__()
        self.flat_size = flat_size
        self.hidden = tuple(hidden)
        self.width = width

        self.spatial = _SpatialInput()
        vec_index = torch.as_tensor(
            field_indices_excluding("arena", "playable_mask"), dtype=torch.long
        )
        self.vec_size = int(vec_index.numel())
        self.norm = _VectorNorm(vec_index)

        ch = self.spatial.channels
        self.trunk = _ConvTrunk(ch, self.vec_size, width)
        self.critic_trunk = _ConvTrunk(ch, self.vec_size, width)

        # Tile logits: one shared 1x1 kernel over the board, plus a learned
        # per-tile bias. The bias is 144 parameters and is what lets the
        # policy say "this exact tile" - the towers are absent from the arena
        # one-hots, so position-specific capacity is not optional here.
        self.tile_conv = nn.Conv2d(width, 1, 1)
        self.tile_bias = nn.Parameter(torch.zeros(TILE_COUNT))

        pooled = width + self.vec_size
        h2 = self.hidden[1]
        self.head_trunk = nn.Sequential(nn.Linear(pooled, h2), nn.ReLU())
        self.play_head = nn.Linear(h2, 2)
        self.slot_head = nn.Linear(h2, HAND_SIZE)
        self.critic_head = nn.Sequential(nn.Linear(pooled, h2), nn.ReLU())
        self.value_head = nn.Linear(h2, 1)

        self.apply(self._init)
        for head, gain in ((self.play_head, 0.01), (self.slot_head, 0.01),
                           (self.tile_conv, 0.01), (self.value_head, 1.0)):
            nn.init.orthogonal_(head.weight.view(head.weight.shape[0], -1), gain)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight.view(m.weight.shape[0], -1), 2 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # The spatial tensor and the normalised vector are built ONCE and fed
        # to both trunks. The trunks stay separate (their gradients must not
        # mix) but the input construction is shared - rebuilding it per trunk
        # doubled the channel assembly for nothing.
        spatial = self.spatial(x)
        vec = self.norm(x)

        z = self.trunk(spatial, vec)
        tile_logits = self.tile_conv(z).view(x.shape[0], TILE_COUNT) + self.tile_bias
        h = self.head_trunk(torch.cat([z.mean(dim=(-2, -1)), vec], dim=-1))

        zc = self.critic_trunk(spatial, vec)
        value = self.value_head(self.critic_head(
            torch.cat([zc.mean(dim=(-2, -1)), vec], dim=-1)
        )).squeeze(-1)

        return self.play_head(h), self.slot_head(h), tile_logits, value

    def actor_parameters(self):
        for m in (self.trunk, self.tile_conv, self.head_trunk,
                  self.play_head, self.slot_head):
            yield from m.parameters()
        yield self.tile_bias

    def critic_parameters(self):
        yield from self.critic_trunk.parameters()
        yield from self.critic_head.parameters()
        yield from self.value_head.parameters()


#: Selectable architectures, by the name ``--arch`` takes and a checkpoint
#: records. A checkpoint with no such field predates conv and is an MLP.
ARCHITECTURES = {"mlp": ActorCritic, "conv": ConvActorCritic}
