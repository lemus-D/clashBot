"""PPO training against the scripted opponent pool.

    python -m src.rl.train --total-steps 2000000 --out models/ppo.pt
    python -m src.rl.train --smoke                     # 30s wiring check
    python -m src.rl.train --eval-only models/ppo.pt   # benchmark a checkpoint

Every run writes a config JSON beside its checkpoint and seeds Python, NumPy
and Torch, so a result can be reproduced rather than remembered.

Checkpoints carry the observation ``schema_hash``. The schema has changed
four times during this project and each change silently reinterprets every
input; refusing to load a mismatched checkpoint is the only thing standing
between that and a policy that appears to work while reading garbage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch

from ..env.observation import schema_descriptor, schema_hash
from ..game.board import HAND_SIZE
from ..sim.env import DeckSpread, LevelSpread, ObservationNoise
from ..sim.opponents import OPPONENTS
from .evaluate import BENCHMARK, evaluate, format_results, headline
from .policy import TILE_COUNT, ActorCritic, to_action
from .ppo import PPO, PPOConfig, Rollout, compute_gae
from .vec_env import VecSimEnv

DEFAULT_OPPONENTS = ("bigspender", "cycler", "tankandsupport", "control")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, net: ActorCritic, cfg: PPOConfig,
                    global_step: int, extra: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": net.state_dict(),
            "flat_size": net.flat_size,
            "hidden": list(net.hidden),
            "schema_hash": schema_hash(),
            "schema": schema_descriptor(),
            "ppo_config": asdict(cfg),
            "global_step": global_step,
            **extra,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    found = ckpt.get("schema_hash")
    if found != schema_hash():
        raise ValueError(
            f"{path} was trained on observation schema {found!r} but this "
            f"code builds {schema_hash()!r}. The channel meanings differ, so "
            f"the weights would be reading the wrong inputs. Retrain, or "
            f"check out the revision that produced it."
        )
    net = ActorCritic(ckpt["flat_size"], tuple(ckpt["hidden"])).to(device)
    net.load_state_dict(ckpt["model_state"])
    return net, ckpt


class MetricLog:
    """One JSON line per update, beside the checkpoint.

    Console output scrolls away and cannot be compared across runs. The
    interesting question after a run is almost never "what was the final
    number" but "WHEN did it stop improving, and what else moved at that
    point" - which needs the whole series, not the last line.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # Truncate: a run owns its log, appending would interleave two runs.
        open(path, "w", encoding="utf-8").close()

    def write(self, kind: str, **fields) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": kind, **fields}) + "\n")


def build_env(args, cfg: PPOConfig) -> VecSimEnv:
    return VecSimEnv(
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        opponents=tuple(args.opponents),
        randomize_scale=0.0 if args.no_randomize else 1.0,
        noise=ObservationNoise.off() if args.no_noise else ObservationNoise(),
        levels=LevelSpread.off() if args.no_levels else LevelSpread(),
        decks=DeckSpread.off() if args.no_decks else DeckSpread(),
    )


def train(args) -> None:
    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        num_envs=args.num_envs,
        lr=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        seed=args.seed,
    )
    seed_everything(cfg.seed)
    device = torch.device(args.device)

    envs = build_env(args, cfg)
    net = ActorCritic(envs.flat_size).to(device)
    algo = PPO(net, cfg, device)

    run_cfg = {
        "ppo": asdict(cfg),
        "opponents": list(args.opponents),
        "schema_hash": schema_hash(),
        "obs_flat_size": envs.flat_size,
        "device": str(device),
        "randomization": {
            "decks": not args.no_decks, "levels": not args.no_levels,
            "noise": not args.no_noise, "stats": not args.no_randomize,
        },
    }
    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(stem + ".config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)
    log = MetricLog(stem + ".metrics.jsonl")
    log.write("config", **run_cfg)
    print(json.dumps(run_cfg, indent=2), flush=True)

    rollout = Rollout(cfg.rollout_steps, cfg.num_envs, envs.flat_size, device,
                      TILE_COUNT, HAND_SIZE)
    obs_np, masks = envs.observe()
    next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)

    global_step = 0
    # Excludes evaluation. Counting eval time against training steps made
    # throughput look ~40x worse than it is, which is the sort of number that
    # gets acted on.
    train_time = 0.0
    recent: list[dict] = []
    best = -1.0

    for update in range(1, cfg.num_updates + 1):
        update_started = time.perf_counter()
        lr = algo.set_lr(1.0 - (update - 1) / cfg.num_updates)

        for step in range(cfg.rollout_steps):
            global_step += cfg.num_envs
            # Normaliser statistics come from the data the policy actually
            # sees, and are frozen at eval so a benchmark is reproducible.
            net.norm.update(next_obs)
            rollout.obs[step] = next_obs
            rollout.done[step] = next_done

            t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
            play, slot, tile, logprob, value, slot_mask, tile_mask = net.act(
                next_obs, t(masks.hand_playable), t(masks.hand_is_spell),
                t(masks.playable),
            )
            rollout.play[step] = play
            rollout.slot[step] = slot
            rollout.tile[step] = tile
            rollout.logprob[step] = logprob
            rollout.value[step] = value
            rollout.slot_mask[step] = slot_mask
            rollout.tile_mask[step] = tile_mask

            actions = [
                to_action(int(p), int(s), int(ti))
                for p, s, ti in zip(play.tolist(), slot.tolist(), tile.tolist())
            ]
            rewards, dones, infos = envs.step(actions)
            rollout.reward[step] = torch.as_tensor(rewards, device=device)

            obs_np, masks = envs.observe()
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(dones, dtype=torch.float32, device=device)

            for info in infos:
                if "episode" in info:
                    recent.append(info["episode"])
            recent = recent[-200:]

        with torch.no_grad():
            last_value = net(next_obs)[3]
        advantages, returns = compute_gae(
            rollout, last_value, next_done, cfg.gamma, cfg.gae_lambda
        )
        stats = algo.update(rollout, advantages, returns)
        # How often the policy actually commits a card. A collapse to
        # all-no-op is the failure mode this action space invites, and it is
        # invisible in reward alone - a passive policy just loses slowly.
        stats["play_rate"] = float(rollout.play.float().mean().item())

        train_time += time.perf_counter() - update_started
        sps = global_step / train_time
        if recent:
            wr = sum(e["result"] == "win" for e in recent) / len(recent)
            ret = float(np.mean([e["return"] for e in recent]))
            ln = float(np.mean([e["length"] for e in recent]))
        else:
            wr = ret = ln = float("nan")
        log.write(
            "update", update=update, step=global_step, sps=sps, lr=lr,
            win_rate=None if np.isnan(wr) else wr,
            mean_return=None if np.isnan(ret) else ret,
            mean_length=None if np.isnan(ln) else ln,
            episodes_seen=len(recent), train_seconds=train_time, **stats,
        )
        print(
            f"upd {update:4d}/{cfg.num_updates}  step {global_step:>9,}  "
            f"{sps:6.0f}/s  lr {lr:.2e}  win {100 * wr:4.0f}%  "
            f"ret {ret:7.2f}  len {ln:5.0f}  "
            f"pl {stats['policy_loss']:+.3f}  vl {stats['value_loss']:.3f}  "
            f"ent {stats['entropy']:.3f}"
            f"[p {stats['entropy_play']:.2f} s {stats['entropy_slot']:.2f} "
            f"t {stats['entropy_tile']:.2f}]  play {stats['play_rate']:.3f}  "
            f"kl {stats['approx_kl']:.4f}  "
            f"ev {stats['explained_variance']:+.2f}",
            flush=True,
        )

        if args.eval_every and update % args.eval_every == 0:
            results = evaluate(net, device, tuple(args.benchmark),
                               episodes=args.eval_episodes)
            print(f"\n  EVAL @ step {global_step:,}")
            print(format_results(results), flush=True)
            score = headline(results)
            log.write("eval", update=update, step=global_step,
                      headline=score, results=[asdict(r) for r in results])
            save_checkpoint(args.out, net, cfg, global_step,
                            {"eval": [asdict(r) for r in results]})
            if score > best:
                best = score
                save_checkpoint(
                    os.path.splitext(args.out)[0] + ".best.pt", net, cfg,
                    global_step, {"eval": [asdict(r) for r in results]},
                )
                print(f"  new best: {100 * score:.0f}%\n", flush=True)
            else:
                print("", flush=True)

    save_checkpoint(args.out, net, cfg, global_step, {})
    print(f"\nsaved {args.out}")
    results = evaluate(net, device, tuple(args.benchmark),
                       episodes=args.eval_episodes)
    print("FINAL")
    print(format_results(results))
    envs.close()


def main() -> None:
    p = argparse.ArgumentParser(description="PPO training in the simulator")
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="credit horizon is ~1/(1-lambda*gamma) STEPS; at the "
                        "default that is ~19 steps against ~700-step episodes")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="models/ppo.pt")
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS),
                   choices=sorted(OPPONENTS))
    p.add_argument("--benchmark", nargs="+", default=list(BENCHMARK),
                   choices=sorted(OPPONENTS))
    p.add_argument("--eval-every", type=int, default=10,
                   help="updates between benchmark runs (0 to disable)")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-only", default=None, metavar="CHECKPOINT",
                   help="benchmark an existing checkpoint and exit")
    p.add_argument("--no-decks", action="store_true")
    p.add_argument("--no-levels", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--no-randomize", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run that exercises the whole loop in seconds")
    args = p.parse_args()

    if args.smoke:
        args.total_steps = 4096
        args.rollout_steps = 32
        args.num_envs = 8
        args.eval_every = 4
        args.eval_episodes = 4
        args.out = args.out.replace(".pt", ".smoke.pt")

    if args.eval_only:
        device = torch.device(args.device)
        net, ckpt = load_checkpoint(args.eval_only, device)
        print(f"{args.eval_only}  step {ckpt.get('global_step', '?'):,}")
        print(format_results(
            evaluate(net, device, tuple(args.benchmark),
                     episodes=args.eval_episodes)
        ))
        return

    train(args)


if __name__ == "__main__":
    main()
