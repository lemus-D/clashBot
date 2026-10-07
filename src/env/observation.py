"""Build a fixed-shape observation dict from board + state.

The observation schema is the contract any ML policy can rely on. All
arrays have known dtypes and shapes regardless of detection noise, so
downstream tensors are stable across frames.

Schema (see ``OBSERVATION_SHAPES`` for the canonical key order):

- ``hand``         (4, V)  one-hot card identity (V = |CARD_CLASSES|)
- ``hand_costs``   (4,)    elixir cost per slot (0 for empty)
- ``hand_playable``(4,)    1.0 where elixir is sufficient, else 0.0
- ``hand_is_spell``(4,)    1.0 where the slot holds a spell
- ``elixir``       float   0-10
- ``match_time``   float   seconds elapsed
- ``time_norm``    float   match_time / MATCH_MAX_DURATION
- ``phase_onehot`` (4,)    normal / double / overtime_double / overtime_triple
- ``arena``        (16, 9, C) one-hot per tile (C = 2 * |ARENA_CLASSES|)
- ``tower_hp``     (6,)    normalized HP per tower (HP-bar fill fraction)
- ``crowns``       (2,)    [friendly, enemy] crown counts
- ``playable_mask``(16, 9) 1 where friendly may place a TROOP

``playable_mask`` is the troop rule only. A spell may be cast anywhere in
the arena, so a slot flagged in ``hand_is_spell`` ignores the mask entirely
- see ``GameBoard.is_placeable``. Four flags carry that instead of four full
144-tile masks, because spell-vs-troop is the only card-dependent rule there
is. Any consumer that masks tile choices MUST resolve the slot first and
then pick the mask, or every legal spell target on the enemy half is
forbidden.

``hand`` and ``arena`` are sized by DIFFERENT class lists - see
``game/classes.py``. Spawn-only units (Goblin Brawler and friends) exist in
the arena but can never be held, so giving the hand a channel for them would
be a permanently dead input.

``flatten`` produces a single 1-D ``np.float32`` vector for MLP-style
policies.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ..game.board import (
    GameBoard,
    HAND_SIZE,
    ARENA_COLS,
    ARENA_ROWS,
)
from ..game.classes import ARENA_CLASSES, CARD_CLASSES, MODEL_ID
from ..game.state import GameState, TOWER_KEYS

PHASES = ("normal", "double", "overtime_double", "overtime_triple")
PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}

# Canonical schema: key order here is the flatten() order. Scalars have
# shape ().
OBSERVATION_SHAPES: dict[str, tuple[int, ...]] = {
    "hand": (HAND_SIZE, len(CARD_CLASSES)),
    "hand_costs": (HAND_SIZE,),
    "hand_playable": (HAND_SIZE,),
    "hand_is_spell": (HAND_SIZE,),
    "elixir": (),
    "match_time": (),
    "time_norm": (),
    "phase_onehot": (len(PHASES),),
    "arena": (ARENA_ROWS, ARENA_COLS, len(ARENA_CLASSES) * 2),
    "tower_hp": (len(TOWER_KEYS),),
    "crowns": (2,),
    "playable_mask": (ARENA_ROWS, ARENA_COLS),
}


def _field_offsets() -> dict[str, tuple[int, int]]:
    """``{field: (offset, size)}`` into the vector ``flatten()`` produces.

    ``flatten()`` concatenates in ``OBSERVATION_SHAPES`` key order, so the
    offsets are derivable rather than magic numbers. The conv policy needs
    them to recover ``arena`` and ``playable_mask`` as 2-D maps from a flat
    observation; hardcoding 67 and 3819 in the network would silently read
    the wrong channels the next time a field is added.
    """
    out: dict[str, tuple[int, int]] = {}
    offset = 0
    for key, shape in OBSERVATION_SHAPES.items():
        size = int(np.prod(shape)) if shape else 1
        out[key] = (offset, size)
        offset += size
    return out


FIELD_OFFSETS: dict[str, tuple[int, int]] = _field_offsets()


def field_indices(*names: str) -> np.ndarray:
    """Flat indices of ``names``, in the order given."""
    return np.concatenate([
        np.arange(FIELD_OFFSETS[k][0], FIELD_OFFSETS[k][0] + FIELD_OFFSETS[k][1])
        for k in names
    ])


def field_indices_excluding(*names: str) -> np.ndarray:
    """Flat indices of every field EXCEPT ``names``, in flatten order."""
    keep = [k for k in OBSERVATION_SHAPES if k not in names]
    return field_indices(*keep)


def schema_descriptor() -> dict:
    """Everything needed to interpret — or later migrate — a flattened
    observation: both class lists (the one-hot channel meanings) and the
    field shapes in flatten order.

    ``model_id`` is recorded too, so a recording says which detector its
    channel meanings came from rather than leaving that to be inferred."""
    return {
        "obs_flat_size": sum(
            int(np.prod(shape)) for shape in OBSERVATION_SHAPES.values()
        ),
        "model_id": MODEL_ID,
        "arena_classes": list(ARENA_CLASSES),
        "card_classes": list(CARD_CLASSES),
        "observation_shapes": {
            k: list(v) for k, v in OBSERVATION_SHAPES.items()
        },
    }


def schema_hash() -> str:
    """Short stable fingerprint of the observation schema."""
    blob = json.dumps(schema_descriptor(), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class ObservationBuilder:
    """Constructs the structured observation dict."""

    def observation_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(OBSERVATION_SHAPES)

    def build(self, board: GameBoard, state: GameState) -> dict:
        elixir = float(state.get_current_elixir())
        match_time = float(state.get_current_match_time())

        phase = state.get_match_phase()
        phase_vec = np.zeros((len(PHASES),), dtype=np.float32)
        if phase in PHASE_INDEX:
            phase_vec[PHASE_INDEX[phase]] = 1.0

        hand_costs = board.hand_costs()
        hand_playable = (hand_costs <= elixir + 1e-6).astype(np.float32)
        hand_playable *= (hand_costs > 0).astype(np.float32)  # empty slot = unplayable

        # Already normalised 0.0-1.0 by the vision layer: it is HP-bar fill,
        # not an HP count, so there is no maximum to divide by here.
        tower_hp = np.array(
            [state.tower_hp[key] for key in TOWER_KEYS], dtype=np.float32
        )
        np.clip(tower_hp, 0.0, 1.0, out=tower_hp)

        playable_mask = board.get_placeable_mask(
            enemy_left_tower_alive=state.is_enemy_left_alive(),
            enemy_right_tower_alive=state.is_enemy_right_alive(),
        ).astype(np.float32)

        return {
            "hand": board.hand_to_tensor(),
            "hand_costs": hand_costs,
            "hand_playable": hand_playable,
            "hand_is_spell": board.hand_is_spell(),
            "elixir": np.float32(elixir),
            "match_time": np.float32(match_time),
            "time_norm": np.float32(match_time / state.MATCH_MAX_DURATION),
            "phase_onehot": phase_vec,
            "arena": board.to_tensor(),
            "tower_hp": tower_hp,
            "crowns": np.array(
                [state.crowns_friendly, state.crowns_enemy], dtype=np.float32
            ),
            "playable_mask": playable_mask,
        }

    @staticmethod
    def flatten(obs: dict) -> np.ndarray:
        """Concatenate all fields into one 1-D float32 vector, in
        ``OBSERVATION_SHAPES`` key order."""
        return np.concatenate(
            [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in OBSERVATION_SHAPES]
        )

    def flat_size(self) -> int:
        return sum(int(np.prod(shape)) for shape in OBSERVATION_SHAPES.values())
