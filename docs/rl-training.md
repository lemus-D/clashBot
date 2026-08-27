# RL training: state of things

Last updated **2026-08-26**, branch `sim-development`.

What this is: the running record of what has been *measured* about training a
policy in the simulator — results, what is settled, what was disproved, and
what the next lead is. `CLAUDE.md` carries the condensed version that gets
loaded every session; this is the longer form with the evidence attached.

---

## 1. Where it stands

`n` is episodes **per opponent**; overall **excludes `idle`**, which anything
functional beats. Full randomization (sampled decks, levels, observation
noise, stat randomization) — the training condition.

| policy      | arch | steps | n   | overall | bigspender | control | cycler | tank+sup |
|-------------|------|-------|-----|---------|-----------|---------|--------|----------|
| random      | —    | —     | 100 | 31%     | 35        | 36      | 27     | 27       |
| `run1`      | mlp  | 819k  | 100 | 44%     | 49        | 45      | 43     | 41       |
| `run2`      | mlp  | 2.0M  | 200 | 49%     | 56        | 51      | 46     | 44       |
| `run3`      | conv | 2.0M  | 200 | **51%** | 56        | 57      | 44     | 47       |
| `run3.best` | conv | 1.97M | 200 | 51%     | 55        | 60      | 46     | 45       |

**Darwin's bar is 60–70%.** Still short.

- Learning over random is **large and real**: ~+19pp, many sigma.
- The **architecture difference is not**: `run3` vs `run2` is 2pp on 800
  scored games each, ≈0.8σ. Treat conv and mlp as **tied**.
- `tankandsupport` is the hardest rung for every policy including random.
  `bigspender` is the easiest for a trained policy but *not* for random,
  where it ties with `control` — punishing a dumped push takes a policy that
  answers it, so that rung only separates once something is learning.

Reproduce any row with:

```
python -m src.rl.train --eval-only models/<ckpt>.pt --eval-episodes 200
python -m src.sim.run --episodes 400 --opponent bigspender,control,cycler,tankandsupport
```

---

## 2. The runs

All three share hyperparameters (`gamma` 0.999, `gae_lambda` 0.99, `seed` 7,
`lr` 3e-4 annealed, `ent_coef` 0.01, 16 envs × 256 rollout = 488 updates at
2M steps), so they are directly comparable.

| run | arch | outcome |
|-----|------|---------|
| `run1` | mlp | Killed at update 206/488. Trained against the broken placement mask. |
| `run2` | mlp | First run ever taken to completion. 488/488, ~700 steps/s, ~50 min. |
| `run3` | conv | Completed. 488/488, ~614 steps/s, ~55 min + evals. |

**`run2` vs `run1`** — training past run1's kill point is worth ~+5pp, which
is only ~1.5σ at n=100. Suggestive, not established. `run2` and `run2.best`
tie, so the last 88 updates and the tail of the LR anneal bought nothing.

**`run3` vs `run2`** — tied (above). Notable secondary findings:

- `run3` **learns slower**: 0% headline at update 40, where the mlp was at
  20%. It is at 1% by update 80 and only reaches the mlp's level around
  update 200.
- `run3` is **less stable**: a collapse to 18% at update 320 that recovered
  by 360. Neither mlp run did anything like that.

---

## 3. What the conv experiment settled

Built to test the standing hypothesis that the ~50% ceiling was
**architectural** — that a `Linear(256, 144)` tile head over a flat trunk
cannot represent spatial structure, because nothing connects tile 37 to tile
38 and every tile must be learned separately.

**It did not unlock placement.** What it actually did:

- Found a **better default tile** — `r12c3`, directly in front of the
  friendly princess towers, against the mlp's `r14c6` behind them — and
  concentrated harder on it.
- **Placement got *less* varied, not more**: 8 distinct tiles used
  deterministically vs the mlp's 21; 72% of placements on one tile vs the
  mlp's 52%.
- Logit margins **shrank**: top1-minus-median legal logit +0.19 vs the mlp's
  +0.92.

Two suspicions checked and **refuted**:

- *The learned per-tile bias is doing all the work, so tile choice is
  effectively input-independent.* No — bias std across tiles 0.029 against
  the conv path's 0.306, and the final argmax never equals the bias's own
  argmax. The head genuinely reads the board.
- *The conv is blind to absolute position.* No — the coordinate channels
  work; the failure is not positional.

**Neither architecture learned context-dependent placement.**

---

## 4. The current best hypothesis

> The ceiling is the **learning signal**, not the function class.

Evidence:

1. A 4.36M-parameter flat MLP and a 137k-parameter conv — 32× apart in size
   and completely different inductive biases — converge on the same ~50%.
2. The **slot head is frozen in both**, at ~95% of maximum entropy. Slot is a
   4-way choice with *no spatial structure at all*, so no architecture can
   explain it. The policy does not learn which card to play any more than it
   learns where to put it.
3. Play rate sits at the sustainable ~0.04 throughout, so this is not no-op
   collapse — it plays about as often as elixir allows, it just does not
   choose well.

**Next lead:** `docs/ideas/placement-shaping.md` — a potential-based spatial
reward gradient, and/or scripted opponents that punish bad placement. Both
still unbuilt. The second is the more attractive framing (it cannot be farmed
and it raises the ceiling rather than the floor) but it changes the frozen
opponent pool, which invalidates every number in section 1 and forces a
re-baseline.

---

## 5. Measurement lessons — read this before quoting a number

### 5.1 Sample size has burned this project three times

Always in the same direction: a promising number shrinking as `n` grew.

| number | at small n | at larger n |
|--------|-----------|-------------|
| `run1` headline | 35% at n=16 | 44% at n=100 |
| `run3` headline | 55% at n=100 | 51% at n=200 |

The SE of the difference between two ~50% rates at n=400 total each is
**≈3.5pp**, so n=400 *cannot resolve* the 2–5pp differences these experiments
actually produce. **Do not quote an architecture or hyperparameter comparison
below n=200/opponent.**

The in-training `--eval-every` passes default to 16 episodes. Those are a
**progress indicator, not a result.**

**Free fix, not implemented:** `evaluate()` already plays every policy on the
same `EVAL_SEED` games, so comparisons are naturally *paired* — but only
aggregate counts are returned, throwing the pairing away. Exporting
per-episode win/loss would make every future comparison substantially more
sensitive at zero extra compute. Do this before the next experiment.

### 5.2 Entropy is the wrong instrument for "did it learn where"

This was recorded as fact in `CLAUDE.md` for several days and was **wrong**.

High tile entropy does *not* mean the tile head learned nothing:

- The benchmark plays **deterministically** (argmax), so logits sitting close
  together still **rank** tiles perfectly well.
- It is partly **enforced by design** — `ent_coef` is applied *unweighted*
  across heads specifically to keep the placement heads exploring, because
  weighting by P(play) gave them ~4% of the bonus and effectively no
  exploration pressure at all.
- Measured on `run3`: tile entropy never moved off 4.44 across the whole run,
  while the tile logits were plainly input-dependent (spread across
  observations 0.21 against 0.31 across tiles).

**Ask the question properly:** measure the **deterministic argmax
distribution** (which tiles get used, how concentrated) and the
**top1-minus-median logit margin**. Those are what drive the benchmark.

Also: the tile entropy ceiling is `ln(~87)` ≈ 4.47, **not** `ln(144)` ≈ 4.97 —
the placement mask only offers ~87 legal tiles on average. Normalising against
144 makes the head look considerably less frozen than it is.

### 5.3 Per-head entropy, not the sum

`entropy_play` / `entropy_slot` / `entropy_tile` are logged to the metrics
JSONL and the console. The **sum hides which head is exploring**: play
collapses to ~1–2% of its maximum while slot sits at ~95% and tile at ~97% of
what its mask allows. A summed 5.7 of 7.05 looks healthy and is not.

---

## 6. Disproved — do not re-run without new evidence

| hypothesis | how it died |
|-----------|-------------|
| `ent_coef` too high | The entropy **gradient** measures 0.000 — at maximum entropy it vanishes. Flat entropy is a *symptom* of a frozen policy, not a cause. |
| Rollout window too short | Quadrupling `rollout_steps` changed nothing. The credit horizon is set by lambda, not the window. |
| Credit horizon too short | Sweeping `gae_lambda` 0.95 → 0.999 (19 → 250 steps) moved KL around noisily and win rate not at all. |
| The placement-mask bug was holding the win rate down | It was a real bug (60% of run1's placements refused, all on one tile) and fixing it moved the headline **not at all** — twice over: re-evaluating run1 under corrected rules, and `run3`/`run2` tracking each other. |
| The tile head can't represent spatial structure | The conv experiment. Tied. See §3. |

The one measurement that **did** pay historically was **gradient norms per
loss term**, which found the critic driving the shared trunk 23:1 and led to
splitting the actor and critic trunks.

The other thing that paid was **measuring the reward distribution** rather
than reasoning about it: before the elixir-trade reward, 91% of steps produced
exactly zero reward and 52% of an episode's whole signal was the win/loss bit
after ~700 decisions. After: 78% silent, terminal down to 25%.

---

## 7. Infrastructure in place

- **`--arch {mlp,conv}`**, default `conv`. Both sit behind a shared
  `FactoredPolicy` base that owns *all* the masking, so there is one
  implementation of the autoregressive slot-then-tile ordering.
- Checkpoints record `arch`; one without the field predates the split and is
  an mlp, so `run1`/`run2` still load.
- **No schema change** — `schema_hash` is `865b69c8f8d4` across every run
  here. The conv reads the arena out of the flat vector via
  `observation.FIELD_OFFSETS`, never a hardcoded offset.
- `board.TOWER_TILES` holds tower tiles for both backends;
  `sim/arena.py` asserts it agrees with its own continuous positions so the
  duplicate cannot drift.
- Per-head entropy in metrics and console.
- `python -m src.sim.run --watch --policy models/run3.pt` is the visual
  debugger, with the reward-split panel.

**Probe scripts** for per-head entropy and deterministic tile sharpness were
written to a session scratchpad and are **not in the repo**. If this line of
work continues they are worth rebuilding as a small `src/rl/diagnose.py` —
they are what produced §3 and §5.2.

---

## 8. If you are picking this up cold

1. Read §1 for where it stands, §5 before quoting any number.
2. The cheapest useful piece of work is the **paired evaluation** in §5.1 —
   it makes everything after it easier to measure.
3. The next real experiment is **reward shaping / punisher opponents**
   (`docs/ideas/placement-shaping.md`), on the §4 hypothesis.
4. Do not re-run anything in §6.
