"""Factored-head imitation network and checkpoint I/O.

One shared MLP trunk, three heads over it:

- ``play`` (1 logit)                    — act this step or not
- ``slot`` (HAND_SIZE logits)           — which card
- ``tile`` (ARENA_ROWS * ARENA_COLS)    — where

The factoring shares every placement across heads: each demo placement
teaches the tile head about locations and the slot head about card
choice independently, instead of splitting examples across 577 flat
classes. The trade-off (slot and tile are chosen independently given
the observation) is documented in the training notes.

Checkpoints carry the network shape and training config so loading
fails loud on schema drift instead of silently mis-slicing.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from ..game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE

HIDDEN_SIZES = (512, 256)


class FactoredPolicyNet(nn.Module):
    def __init__(self, flat_size: int, hidden: tuple[int, int] = HIDDEN_SIZES):
        super().__init__()
        self.flat_size = flat_size
        self.hidden = tuple(hidden)
        h1, h2 = self.hidden
        self.trunk = nn.Sequential(
            nn.Linear(flat_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.play_head = nn.Linear(h2, 1)
        self.slot_head = nn.Linear(h2, HAND_SIZE)
        self.tile_head = nn.Linear(h2, ARENA_ROWS * ARENA_COLS)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (play_logit (B,), slot_logits (B,4), tile_logits (B,144))."""
        z = self.trunk(x)
        return (
            self.play_head(z).squeeze(-1),
            self.slot_head(z),
            self.tile_head(z),
        )


def save_checkpoint(
    path: str, model: FactoredPolicyNet, config: dict[str, Any]
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "flat_size": model.flat_size,
            "hidden": list(model.hidden),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str, device: str
) -> tuple[FactoredPolicyNet, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = FactoredPolicyNet(ckpt["flat_size"], tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["config"]
