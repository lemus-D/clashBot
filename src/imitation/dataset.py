"""Load recorded two-stream JSONL demos into arrays for training.

Recordings hold two independent timestamped streams (see
``src/env/environment.py``): ``{"type": "obs"}`` lines from the
perception loop and ``{"type": "act"}`` lines from the bot's executor or
the human's mouse. This module joins them. Any other line type (e.g.
``{"type": "diag"}`` lifecycle diagnostics) is skipped.

Pairing rule: **every action attaches to the nearest observation whose
timestamp strictly precedes it** — the last state the actor could
actually have seen when it acted. Consequences:

- An observation with N attached actions yields N training rows (same
  observation, different action targets). A ~0.3s perception cycle
  easily spans two or three human placements, and all of them are real
  demonstrations of "act on this state".
- An observation with no attached action yields one NO_OP row.
- Actions timestamped before the first observation of their file have no
  state to attach to and are dropped (counted and reported).
- Pairing is by timestamp, not file order: an action can be flushed to
  disk a cycle late and still land on the correct earlier observation.
- Ties (same 15.6ms wall-clock tick as an observation) resolve to the
  observation *before* it; see ``_pair``.

Each row's action decomposes into three labels for the factored heads:

- ``play``  (float 0/1)  — was a card placed on this observation
- ``slot``  (int 0-3)    — which hand slot (only where play == 1)
- ``tile``  (int 0-143)  — tile_y * ARENA_COLS + tile_x (ditto)

No-ops vastly outnumber placements: perception runs at roughly 3 cycles
per second, so a full match is ~900 observations against maybe 20-40
placements — a 20:1 to 45:1 imbalance. ``load_demos`` downsamples no-op
rows to ``no_op_ratio`` times the placement count; placement rows are
always kept.
"""

from __future__ import annotations

import bisect
import json

import numpy as np

from ..env.environment import check_record_meta
from ..env.observation import ObservationBuilder, schema_hash
from ..game.board import ARENA_COLS


def _read_streams(path: str, flat_size: int) -> tuple[list[dict], list[dict]]:
    """Parse one recording into (observations, actions), both sorted by
    timestamp. Validates the meta header and observation width."""
    meta_seen = False
    observations: list[dict] = []
    actions: list[dict] = []

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("type")

            if kind == "meta":
                check_record_meta(rec, path)
                if rec["schema_hash"] != schema_hash():
                    raise ValueError(
                        f"{path} was recorded under observation schema "
                        f"{rec['schema_hash']} but the current schema "
                        f"is {schema_hash()}. Migrate it first — its "
                        f"meta header retains the troop class list "
                        f"and field shapes it was recorded with."
                    )
                meta_seen = True
                continue

            if not meta_seen:
                raise ValueError(
                    f"{path}:{lineno}: record before the meta header. "
                    f"Recordings must start with a "
                    f'{{"type": "meta", ...}} line.'
                )

            if kind == "obs":
                flat = rec["obs_flat"]
                if len(flat) != flat_size:
                    raise ValueError(
                        f"{path}:{lineno}: obs_flat has {len(flat)} values, "
                        f"expected {flat_size}. This demo was recorded with "
                        f"a different observation schema and cannot be mixed "
                        f"with the current one."
                    )
                observations.append(rec)
            elif kind == "act":
                actions.append(rec)
            # Anything else is an auxiliary stream this trainer has no use
            # for — currently {"type": "diag"} lifecycle diagnostics, which
            # every --debug run writes. Skipped, not an error: auxiliary
            # line types are additive by design (they leave obs/act lines
            # untouched, hence no record_format bump), so refusing them
            # would break training on files that are perfectly valid. The
            # strictness that protects the data is still enforced above:
            # the meta header, the schema hash and the obs_flat width.

    if not meta_seen:
        raise ValueError(
            f"{path} has no meta header; it predates schema versioning "
            f"and cannot be read."
        )

    observations.sort(key=lambda r: r["t"])
    actions.sort(key=lambda r: r["t"])
    return observations, actions


def _pair(
    observations: list[dict], actions: list[dict]
) -> tuple[list[list[dict]], int]:
    """Attach each action to the nearest preceding observation.

    Returns a list parallel to ``observations`` holding the actions bound
    to each, plus the number of actions dropped for having no preceding
    observation.
    """
    times = [r["t"] for r in observations]
    attached: list[list[dict]] = [[] for _ in observations]
    orphans = 0
    for act in actions:
        # bisect_left => *strictly* preceding. Windows ``time.time()`` has
        # ~15.6ms granularity, so an action released just before a frame
        # grab can land in the same tick as it. Reaction time is never
        # zero, so a tie resolves backwards: the actor cannot have been
        # responding to a frame captured in the same instant.
        i = bisect.bisect_left(times, act["t"]) - 1
        if i < 0:
            orphans += 1
            continue
        attached[i].append(act)
    return attached, orphans


def load_demos(
    paths: list[str],
    no_op_ratio: float = 3.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Parse demo JSONL files and return downsampled training arrays.

    Returns ``{"obs": (N, flat_size) float32, "play": (N,) float32,
    "slot": (N,) int64, "tile": (N,) int64}``. Raises on empty input, on
    demos with zero placements, on an unreadable record format, and on
    ``obs_flat`` rows whose length doesn't match the current schema.
    """
    flat_size = ObservationBuilder().flat_size()
    obs_rows: list[list[float]] = []
    play: list[float] = []
    slot: list[int] = []
    tile: list[int] = []
    orphans = 0

    for path in paths:
        observations, actions = _read_streams(path, flat_size)
        attached, dropped = _pair(observations, actions)
        orphans += dropped

        for obs, acts in zip(observations, attached):
            if not acts:
                obs_rows.append(obs["obs_flat"])
                play.append(0.0)
                slot.append(0)
                tile.append(0)
                continue
            for act in acts:
                obs_rows.append(obs["obs_flat"])
                play.append(1.0)
                slot.append(act["hand_index"])
                tile.append(act["tile_y"] * ARENA_COLS + act["tile_x"])

    if not obs_rows:
        raise ValueError(f"No observations found in {paths}")

    obs_arr = np.asarray(obs_rows, dtype=np.float32)
    play_arr = np.asarray(play, dtype=np.float32)
    slot_arr = np.asarray(slot, dtype=np.int64)
    tile_arr = np.asarray(tile, dtype=np.int64)

    placement_idx = np.flatnonzero(play_arr == 1.0)
    noop_idx = np.flatnonzero(play_arr == 0.0)
    if len(placement_idx) == 0:
        raise ValueError(
            f"Demos in {paths} contain {len(play_arr)} observations but zero "
            f"placements; nothing to imitate. Was the mouse listener able "
            f"to see your drags?"
        )

    if orphans:
        print(
            f"WARNING: dropped {orphans} action(s) timestamped before the "
            f"first observation of their file"
        )

    rng = np.random.default_rng(seed)
    n_keep = min(len(noop_idx), int(no_op_ratio * len(placement_idx)))
    kept_noops = rng.choice(noop_idx, size=n_keep, replace=False)
    idx = np.sort(np.concatenate([placement_idx, kept_noops]))

    print(
        f"Loaded {len(play_arr)} rows ({len(placement_idx)} placements); "
        f"kept {n_keep}/{len(noop_idx)} no-ops "
        f"(ratio {no_op_ratio:g}) -> {len(idx)} training rows"
    )
    return {
        "obs": obs_arr[idx],
        "play": play_arr[idx],
        "slot": slot_arr[idx],
        "tile": tile_arr[idx],
    }
