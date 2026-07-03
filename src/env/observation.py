"""Build a fixed-shape observation dict from board + state.

The observation schema is the contract any ML policy can rely on. All
arrays have known dtypes and shapes regardless of detection noise, so
downstream tensors are stable across frames.

Schema (see ``OBSERVATION_SHAPES`` for the canonical key order):

- ``hand``         (4, V)  one-hot card identity (V = |TROOP_CLASSES|)
- ``hand_costs``   (4,)    elixir cost per slot (0 for empty)
- ``hand_playable``(4,)    1.0 where elixir is sufficient, else 0.0
- ``elixir``       float   0-10
- ``match_time``   float   seconds elapsed
- ``time_norm``    float   match_time / MATCH_MAX_DURATION
- ``phase_onehot`` (4,)    normal / double / overtime_double / overtime_triple
- ``arena``        (16, 9, C) one-hot per tile (C = 2 * |TROOP_CLASSES|)
- ``tower_hp``     (6,)    normalized HP per tower
- ``crowns``       (2,)    [friendly, enemy] crown counts
- ``playable_mask``(16, 9) 1 where friendly may place

``flatten`` produces a single 1-D ``np.float32`` vector for MLP-style
policies.
"""

from __future__ import annotations

import numpy as np

from ..game.board import (
    GameBoard,
    HAND_SIZE,
    ARENA_COLS,
    ARENA_ROWS,
    TROOP_CLASSES,
)
from ..game.state import GameState, TOWER_KEYS

PHASES = ("normal", "double", "overtime_double", "overtime_triple")
PHASE_INDEX = {p: i for i, p in enumerate(PHASES)}

# Canonical schema: key order here is the flatten() order. Scalars have
# shape ().
OBSERVATION_SHAPES: dict[str, tuple[int, ...]] = {
    "hand": (HAND_SIZE, len(TROOP_CLASSES)),
    "hand_costs": (HAND_SIZE,),
    "hand_playable": (HAND_SIZE,),
    "elixir": (),
    "match_time": (),
    "time_norm": (),
    "phase_onehot": (len(PHASES),),
    "arena": (ARENA_ROWS, ARENA_COLS, len(TROOP_CLASSES) * 2),
    "tower_hp": (len(TOWER_KEYS),),
    "crowns": (2,),
    "playable_mask": (ARENA_ROWS, ARENA_COLS),
}


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

        tower_hp = np.zeros((len(TOWER_KEYS),), dtype=np.float32)
        for i, key in enumerate(TOWER_KEYS):
            current = state.tower_hp[key]
            if current is None:
                tower_hp[i] = 1.0
            else:
                tower_hp[i] = float(
                    np.clip(current / state.tower_max_hp[key], 0.0, 1.0)
                )

        playable_mask = board.get_placeable_mask(
            enemy_left_tower_alive=state.is_enemy_left_alive(),
            enemy_right_tower_alive=state.is_enemy_right_alive(),
            enemy_king_active=state.is_enemy_king_active(),
        ).astype(np.float32)

        return {
            "hand": board.hand_to_tensor(),
            "hand_costs": hand_costs,
            "hand_playable": hand_playable,
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
