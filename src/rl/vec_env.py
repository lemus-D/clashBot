"""A batch of ``SimEnv`` instances stepped together.

Synchronous on purpose. The simulator is pure Python, so this does NOT run
the envs in parallel - what it buys is one batched network forward pass per
step instead of N, which is the larger cost once the batch is wide. At ~400x
real time per env, a synchronous batch of 32 is already far past anything the
hardware pipeline could produce, and true multiprocessing can come later if
the sim ever becomes the bottleneck again.

Envs AUTO-RESET: when one finishes, its terminal observation is reported in
``info`` and the observation returned is the first of the next episode. That
keeps every env producing experience every step, which a fixed-length PPO
rollout needs.

Each env gets its own opponent instance. Sharing one would let a bot's
internal state - TankAndSupport's committed lane, a reaction timer - be
driven by 32 different matches at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env.actions import Action
from ..env.observation import ObservationBuilder
from ..game.board import ARENA_COLS, ARENA_ROWS, HAND_SIZE
from ..sim.env import DeckSpread, LevelSpread, ObservationNoise, SimEnv
from ..sim.opponents import make_opponent

TILE_COUNT = ARENA_ROWS * ARENA_COLS


@dataclass
class Masks:
    """The three arrays the policy needs to mask illegal actions."""

    hand_playable: np.ndarray   # (N, 4)
    hand_is_spell: np.ndarray   # (N, 4)
    playable: np.ndarray        # (N, 144)


@dataclass
class VecSimEnv:
    """``num_envs`` simulators stepped in lockstep."""

    num_envs: int = 16
    seed: int = 0
    opponents: tuple[str, ...] = ("cycler",)
    randomize_scale: float = 1.0
    noise: ObservationNoise = field(default_factory=ObservationNoise)
    levels: LevelSpread = field(default_factory=LevelSpread)
    decks: DeckSpread = field(default_factory=DeckSpread)
    #: Opponent-only deck sampler; None means "same as ``decks``", which
    #: is what the frozen pool's recorded numbers assume.
    opponent_decks: object | None = None

    def __post_init__(self) -> None:
        self.envs: list[SimEnv] = []
        self._opponent_names: list[str] = []
        for i in range(self.num_envs):
            # Round-robin rather than random so a rollout always contains
            # every opponent in the mix, whatever the batch size.
            name = self.opponents[i % len(self.opponents)]
            self._opponent_names.append(name)
            self.envs.append(SimEnv(
                seed=self.seed + i * 1000,
                opponent=make_opponent(name, seed=self.seed + i),
                randomize_scale=self.randomize_scale,
                noise=self.noise,
                levels=self.levels,
                decks=self.decks,
                opponent_decks=self.opponent_decks,
            ))
        self._obs = [env.reset() for env in self.envs]
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    # ----- observation -----

    @property
    def flat_size(self) -> int:
        return ObservationBuilder().flat_size()

    def observe(self) -> tuple[np.ndarray, Masks]:
        flat = np.stack([ObservationBuilder.flatten(o) for o in self._obs])
        masks = Masks(
            hand_playable=np.stack([o["hand_playable"] for o in self._obs]),
            hand_is_spell=np.stack([o["hand_is_spell"] for o in self._obs]),
            playable=np.stack(
                [np.asarray(o["playable_mask"]).reshape(-1) for o in self._obs]
            ),
        )
        return flat, masks

    # ----- stepping -----

    def step(self, actions: list[Action]):
        """Step every env once. Returns ``(rewards, dones, infos)``.

        ``infos`` carries ``episode`` (return, length, result, opponent) only
        on the step an episode ends, so a logger can pick out completions
        without tracking state of its own.
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(action)
            rewards[i] = reward
            dones[i] = done
            self.episode_returns[i] += reward
            self.episode_lengths[i] += 1

            if done:
                info = dict(info)
                crowns = info.get("crowns", (0, 0))
                info["episode"] = {
                    "return": float(self.episode_returns[i]),
                    "length": int(self.episode_lengths[i]),
                    "result": info.get("result"),
                    "crowns_for": int(crowns[0]),
                    "crowns_against": int(crowns[1]),
                    "opponent": self._opponent_names[i],
                }
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0
                obs = env.reset()

            self._obs[i] = obs
            infos.append(info)

        return rewards, dones, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
