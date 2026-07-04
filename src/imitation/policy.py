"""Run a trained factored-head checkpoint as a ``policy(obs) -> Action``.

Decision procedure per step:

1. ``sigmoid(play_logit) < play_threshold`` -> NO_OP. The threshold is
   the aggressiveness knob: raising it makes the bot hold cards longer.
2. Slot logits are masked by ``obs["hand_playable"]`` (empty slot or
   insufficient elixir), tile logits by ``obs["playable_mask"]``; each
   is argmaxed. If every slot or every tile is masked -> NO_OP.

Masking at inference matters more than the model with small datasets:
the net will happily rank an unaffordable card first, and the mask
recovers that for free.
"""

from __future__ import annotations

import numpy as np
import torch

from ..env.actions import Action
from ..env.observation import ObservationBuilder, schema_hash
from ..game.board import ARENA_COLS
from .model import load_checkpoint


class ImitationPolicy:
    def __init__(
        self,
        weights_path: str,
        play_threshold: float = 0.5,
        device: str | None = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.play_threshold = play_threshold
        self.model, self.train_config = load_checkpoint(
            weights_path, self.device
        )

        expected = ObservationBuilder().flat_size()
        if self.model.flat_size != expected:
            raise ValueError(
                f"Checkpoint {weights_path!r} was trained on "
                f"{self.model.flat_size}-dim observations but the current "
                f"schema is {expected}-dim; retrain on demos recorded with "
                f"this schema."
            )
        trained_hash = self.train_config.get("schema_hash")
        if trained_hash is not None and trained_hash != schema_hash():
            raise ValueError(
                f"Checkpoint {weights_path!r} was trained under observation "
                f"schema {trained_hash} but the current schema is "
                f"{schema_hash()}; retrain on demos recorded with this "
                f"schema."
            )

    def __call__(self, obs: dict) -> Action:
        flat = ObservationBuilder.flatten(obs)
        x = torch.from_numpy(flat).to(self.device).unsqueeze(0)
        with torch.no_grad():
            play_logit, slot_logits, tile_logits = self.model(x)

        if torch.sigmoid(play_logit).item() < self.play_threshold:
            return Action.no_op()

        slot_mask = np.asarray(obs["hand_playable"]) > 0
        tile_mask = np.asarray(obs["playable_mask"]).reshape(-1) > 0
        if not slot_mask.any() or not tile_mask.any():
            return Action.no_op()

        slot_np = slot_logits[0].cpu().numpy()
        tile_np = tile_logits[0].cpu().numpy()
        slot_np[~slot_mask] = -np.inf
        tile_np[~tile_mask] = -np.inf

        slot = int(slot_np.argmax())
        tile_y, tile_x = divmod(int(tile_np.argmax()), ARENA_COLS)
        return Action(hand_index=slot, tile_x=tile_x, tile_y=tile_y)
