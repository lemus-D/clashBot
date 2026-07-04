"""Load recorded JSONL demos into arrays for factored-head training.

Each record's action decomposes into three labels:

- ``play``  (float 0/1)  — was a card placed this step (hand_index >= 0)
- ``slot``  (int 0-3)    — which hand slot (only meaningful where play == 1)
- ``tile``  (int 0-143)  — tile_y * ARENA_COLS + tile_x (ditto)

No-op steps vastly outnumber placements (~20:1 at 4 steps/sec), so
``load_demos`` downsamples them to ``no_op_ratio`` times the placement
count. Placement rows are always kept.
"""

from __future__ import annotations

import json

import numpy as np

from ..env.observation import ObservationBuilder, schema_hash
from ..game.board import ARENA_COLS


def load_demos(
    paths: list[str],
    no_op_ratio: float = 3.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Parse demo JSONL files and return downsampled training arrays.

    Returns ``{"obs": (N, flat_size) float32, "play": (N,) float32,
    "slot": (N,) int64, "tile": (N,) int64}``. Raises on empty input,
    on demos with zero placements, and on ``obs_flat`` rows whose
    length doesn't match the current observation schema.
    """
    flat_size = ObservationBuilder().flat_size()
    obs_rows: list[list[float]] = []
    play: list[float] = []
    slot: list[int] = []
    tile: list[int] = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "meta":
                    if rec["schema_hash"] != schema_hash():
                        raise ValueError(
                            f"{path} was recorded under observation schema "
                            f"{rec['schema_hash']} but the current schema "
                            f"is {schema_hash()}. Migrate it first — its "
                            f"meta header retains the troop class list "
                            f"and field shapes it was recorded with."
                        )
                    continue
                flat = rec["obs_flat"]
                if len(flat) != flat_size:
                    raise ValueError(
                        f"{path}:{lineno}: obs_flat has {len(flat)} values, "
                        f"expected {flat_size}. This demo was recorded with "
                        f"a different observation schema and cannot be mixed "
                        f"with the current one."
                    )
                is_play = rec["hand_index"] >= 0
                obs_rows.append(flat)
                play.append(1.0 if is_play else 0.0)
                slot.append(rec["hand_index"] if is_play else 0)
                tile.append(
                    rec["tile_y"] * ARENA_COLS + rec["tile_x"] if is_play else 0
                )

    if not obs_rows:
        raise ValueError(f"No records found in {paths}")

    obs_arr = np.asarray(obs_rows, dtype=np.float32)
    play_arr = np.asarray(play, dtype=np.float32)
    slot_arr = np.asarray(slot, dtype=np.int64)
    tile_arr = np.asarray(tile, dtype=np.int64)

    placement_idx = np.flatnonzero(play_arr == 1.0)
    noop_idx = np.flatnonzero(play_arr == 0.0)
    if len(placement_idx) == 0:
        raise ValueError(
            f"Demos in {paths} contain {len(play_arr)} steps but zero "
            f"placements; nothing to imitate. Was the mouse listener able "
            f"to see your drags?"
        )

    rng = np.random.default_rng(seed)
    n_keep = min(len(noop_idx), int(no_op_ratio * len(placement_idx)))
    kept_noops = rng.choice(noop_idx, size=n_keep, replace=False)
    idx = np.sort(np.concatenate([placement_idx, kept_noops]))

    print(
        f"Loaded {len(play_arr)} steps ({len(placement_idx)} placements); "
        f"kept {n_keep}/{len(noop_idx)} no-ops "
        f"(ratio {no_op_ratio:g}) -> {len(idx)} training rows"
    )
    return {
        "obs": obs_arr[idx],
        "play": play_arr[idx],
        "slot": slot_arr[idx],
        "tile": tile_arr[idx],
    }
