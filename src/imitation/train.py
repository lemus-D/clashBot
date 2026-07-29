"""Train the factored-head imitation policy from recorded demos.

Usage::

    python -m src.imitation.train demos/run.jsonl --out models/imitation.pt
    python -m src.imitation.train demos/*.jsonl --epochs 50 --no-op-ratio 2

Loss = BCE(play, all rows) + CE(slot) + CE(tile), the latter two only
on rows where a card was placed. Runs on CUDA when available. The run
config is printed and stored inside the checkpoint for reproducibility.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from ..env.observation import schema_hash
from .dataset import load_demos
from .model import FactoredPolicyNet, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train imitation policy")
    p.add_argument("demos", nargs="+", help="demo JSONL file(s)")
    p.add_argument("--out", required=True, help="checkpoint output path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-op-ratio", type=float, default=3.0,
                   help="kept no-op rows per placement row")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def batch_loss(
    model: FactoredPolicyNet,
    obs: torch.Tensor,
    play: torch.Tensor,
    slot: torch.Tensor,
    tile: torch.Tensor,
) -> torch.Tensor:
    play_logit, slot_logits, tile_logits = model(obs)
    loss = F.binary_cross_entropy_with_logits(play_logit, play)
    placed = play > 0.5
    if placed.any():
        loss = loss + F.cross_entropy(slot_logits[placed], slot[placed])
        loss = loss + F.cross_entropy(tile_logits[placed], tile[placed])
    return loss


@torch.no_grad()
def evaluate(model: FactoredPolicyNet, tensors: dict[str, torch.Tensor]) -> str:
    play_logit, slot_logits, tile_logits = model(tensors["obs"])
    play_acc = (
        ((play_logit > 0) == (tensors["play"] > 0.5)).float().mean().item()
    )
    placed = tensors["play"] > 0.5
    if placed.any():
        slot_acc = (
            (slot_logits[placed].argmax(-1) == tensors["slot"][placed])
            .float().mean().item()
        )
        tile_acc = (
            (tile_logits[placed].argmax(-1) == tensors["tile"][placed])
            .float().mean().item()
        )
        return (
            f"play_acc={play_acc:.3f} slot_acc={slot_acc:.3f} "
            f"tile_acc={tile_acc:.3f}"
        )
    return f"play_acc={play_acc:.3f} (no placements in val split)"


def train() -> None:
    args = parse_args()
    config = vars(args) | {"schema_hash": schema_hash()}
    print(f"Config: {config}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = load_demos(args.demos, no_op_ratio=args.no_op_ratio, seed=args.seed)
    n = len(data["play"])
    perm = np.random.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    splits = {"val": perm[:n_val], "train": perm[n_val:]}
    tensors = {
        name: {
            "obs": torch.from_numpy(data["obs"][idx]).to(device),
            "play": torch.from_numpy(data["play"][idx]).to(device),
            "slot": torch.from_numpy(data["slot"][idx]).to(device),
            "tile": torch.from_numpy(data["tile"][idx]).to(device),
        }
        for name, idx in splits.items()
    }
    n_train = len(splits["train"])
    print(f"Split: {n_train} train / {n_val} val")

    model = FactoredPolicyNet(flat_size=data["obs"].shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n_train, device=device)
        total = 0.0
        for start in range(0, n_train, args.batch_size):
            b = order[start:start + args.batch_size]
            t = tensors["train"]
            loss = batch_loss(
                model, t["obs"][b], t["play"][b], t["slot"][b], t["tile"][b]
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(b)

        model.eval()
        print(
            f"epoch {epoch:3d}/{args.epochs} "
            f"train_loss={total / n_train:.4f} "
            f"val: {evaluate(model, tensors['val'])}"
        )

    save_checkpoint(args.out, model, config)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    train()
