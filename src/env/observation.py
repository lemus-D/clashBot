"""Build a fixed-shape observation dict from board + state.

The observation schema is the contract any ML policy can rely on. All
arrays have known dtypes and shapes regardless of detection noise, so
downstream tensors are stable across frames.

Schema produced by ``build``:

- ``hand``         (4, V)  one-hot card identity
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

from typing import Optional

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


class ObservationBuilder:
    """Constructs the structured observation dict.

    ``card_vocab`` defaults to ``TROOP_CLASSES`` so hand and arena share
    a vocabulary; pass an explicit vocab if you want to widen it for
    spell cards.
    """

    def __init__(self, card_vocab: Optional[list[str]] = None):
        self.card_vocab = list(card_vocab) if card_vocab is not None else list(TROOP_CLASSES)
        self.vocab_size = len(self.card_vocab)
        self.arena_channels = len(TROOP_CLASSES) * 2

    # ----- shape inspection -----

    def observation_shapes(self) -> dict[str, tuple]:
        return {
            "hand": (HAND_SIZE, self.vocab_size),
            "hand_costs": (HAND_SIZE,),
            "hand_playable": (HAND_SIZE,),
            "elixir": (),
            "match_time": (),
            "time_norm": (),
            "phase_onehot": (len(PHASES),),
            "arena": (ARENA_ROWS, ARENA_COLS, self.arena_channels),
            "tower_hp": (len(TOWER_KEYS),),
            "crowns": (2,),
            "playable_mask": (ARENA_ROWS, ARENA_COLS),
        }

    # ----- main entry point -----

    def build(self, board: GameBoard, state: GameState) -> dict:
        elixir = float(state.get_current_elixir())
        match_time = float(state.get_current_match_time())

        phase = state.get_match_phase()
        phase_vec = np.zeros((len(PHASES),), dtype=np.float32)
        if phase in PHASE_INDEX:
            phase_vec[PHASE_INDEX[phase]] = 1.0

        hand = board.hand_to_tensor(self.card_vocab)
        hand_costs = board.hand_costs()
        hand_playable = (hand_costs <= elixir + 1e-6).astype(np.float32)
        hand_playable *= (hand_costs > 0).astype(np.float32)  # empty slot = unplayable

        tower_hp = np.zeros((len(TOWER_KEYS),), dtype=np.float32)
        for i, key in enumerate(TOWER_KEYS):
            current = state.tower_hp.get(key)
            max_hp = state.tower_max_hp.get(key, 1) or 1
            if current is None:
                tower_hp[i] = 1.0
            else:
                tower_hp[i] = float(np.clip(current / max_hp, 0.0, 1.0))

        crowns = np.array(
            [state.crowns_friendly, state.crowns_enemy], dtype=np.float32
        )

        playable_mask = board.get_placeable_mask(
            enemy_left_tower_alive=state.is_enemy_left_alive(),
            enemy_right_tower_alive=state.is_enemy_right_alive(),
            enemy_king_active=state.is_enemy_king_active(),
        ).astype(np.float32)

        return {
            "hand": hand.astype(np.float32),
            "hand_costs": hand_costs.astype(np.float32),
            "hand_playable": hand_playable,
            "elixir": np.float32(elixir),
            "match_time": np.float32(match_time),
            "time_norm": np.float32(match_time / state.MATCH_MAX_DURATION),
            "phase_onehot": phase_vec,
            "arena": board.to_tensor(),
            "tower_hp": tower_hp,
            "crowns": crowns,
            "playable_mask": playable_mask,
        }

    # ----- flat representation -----

    @staticmethod
    def flatten(obs: dict) -> np.ndarray:
        parts = []
        for key in (
            "hand",
            "hand_costs",
            "hand_playable",
            "elixir",
            "match_time",
            "time_norm",
            "phase_onehot",
            "arena",
            "tower_hp",
            "crowns",
            "playable_mask",
        ):
            v = obs[key]
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            parts.append(arr)
        return np.concatenate(parts, axis=0)

    def flat_size(self) -> int:
        size = 0
        for shape in self.observation_shapes().values():
            if not shape:
                size += 1
            else:
                n = 1
                for d in shape:
                    n *= d
                size += n
        return size
