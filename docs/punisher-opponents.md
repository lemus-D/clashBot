# Punisher opponents: making placement matter

Branch `punisher-opponents`. Companion to `docs/rl-training.md`, which carries
the training record; this page carries the opponent design and what was
measured building it.

---

## 1. Why these exist

`docs/rl-training.md` §4 concludes that the ~50% ceiling is the **learning
signal**, not the function class: a 4.36M-parameter MLP and a 137k-parameter
conv converge on the same number, and the *slot* head — a 4-way choice with no
spatial structure at all — is frozen in both at ~95% of maximum entropy.

One reason a placement signal might be absent is simply that **placement does
not change the outcome**. The four original scripted bots pick their lane with
`rng.choice(LANES)` and answer whatever walks into their half. Against an
opponent like that, a policy that plays every card on one favourite tile scores
about the same as a policy that places well — so no reward derived from the
outcome can teach placement, however it is shaped.

These two bots are an attempt to close that gap. They read where the policy
placed and exploit it. They are the "punisher opponents" half of
`docs/ideas/placement-shaping.md`; the potential-based reward-shaping half is
**still unbuilt**.

> **Status: MEASURED AND NEGATIVE.** They are harder - `punisher` holds a
> random policy to 18.0% where the hardest frozen bot allows 27.0% - but §4
> shows they do NOT make placement matter more. Across four fixed tiles the
> outcome range is 14.7pp against them versus 21.3pp against the frozen four,
> so the design goal did not happen. Use the pool as a harder yardstick; do
> not cite it as the answer to §4 of the training doc.
>
> §4 also turned up a lead worth more than the pool: concentrating on one tile
> BEATS spreading in this simulator, on three of four tiles tried. If that
> holds, `run3`'s concentrated placement is rational rather than broken.

---

## 2. The two pools, and why they are separate

| pool | members | status |
|------|---------|--------|
| `BASELINE_POOL` | `bigspender`, `control`, `cycler`, `tankandsupport` | **FROZEN.** Byte-identical to what §1's numbers were measured against. |
| `PUNISHER_POOL` | `punisher`, `controlplus` | New. No policy baseline yet. |

Nothing in the original four was edited. `controlplus` is a **separate bot**
rather than a change to `control` for exactly this reason: the scripted pool is
the only absolute scale this project has, and a win rate scored against a
changed pool is not comparable to a recorded one however similar the bots look.

Re-measured after all of this, a random policy still scores **31.0%** overall
against the frozen four (n=100/opponent, seed 777) — the same 31% §1 records.
That is the evidence the freeze held.

### How hard they are

Random policy, n=200 each, seed 777. **Higher means a weaker opponent.**

| opponent | random wins |
|----------|-------------|
| `bigspender` (frozen) | 40.0%* |
| `control` (frozen) | 33.0% |
| `tankandsupport` (frozen) | 31.0% |
| `cycler` (frozen) | 27.0%* |
| `controlplus` | **29.5%** |
| `controlplus` + structured decks | 31.5% |
| `punisher` | **18.0%** |
| `punisher` + structured decks | 24.5% |

\* measured at n=100 in the same run that reproduced the 31.0% headline.

`punisher` is comfortably the hardest opponent in the project — 9pp clear of
`cycler`, the hardest frozen bot, which is well outside the ~4.4pp SE of a
difference at this n. `controlplus` is a smaller step past the bot it
sharpens (29.5% against `control`'s 33.0%, about 1σ — suggestive, not
established).

Structured decks cost both bots (18.0 → 24.5 and 29.5 → 31.5). See §6.

```
python -m src.sim.run --episodes 400 --opponent baseline
python -m src.sim.run --episodes 400 --opponent punishers --structured-decks
python -m src.rl.train --opponents punishers --structured-decks --benchmark punishers
```

---

## 3. What they actually punish

### 3.1 Clumping, via spells

Stacking units on one tile is precisely what a policy with a concentrated tile
head does — `run3` put 72% of its placements on a single tile. A Fireball on
the stack takes the whole push for four elixir.

`OpponentView.spell_targets` scans candidate centres at each enemy unit (an
optimal centre always sits inside the cluster it hits, so sweeping all 144
tiles buys nothing) and ranks them by **elixir caught**.

### 3.2 Lane imbalance, via where the push goes

`PlacementAware.target_lane` prefers the lane the enemy has committed *less*
elixir to, when the split exceeds `LANE_IMBALANCE_ELIXIR`; otherwise it falls
back to whichever enemy princess tower is closest to falling. Both beat
`rng.choice(LANES)`, which is what every frozen bot does.

Concentrating on one tower is also just how matches are closed. Two towers at
50% is a draw; one at 0% is a crown.

### 3.3 Not over-answering

`MIN_THREAT_TO_ANSWER` makes them decline to spend a card on a push worth less
than 3 elixir. A policy that dribbles single cheap units gets nothing back.

---

## 4. Do they make placement matter? NO

**Read this before quoting anything above as a success.**

These bots are *harder*. That was never the goal — the goal was to make
placement change the outcome, so a reward derived from the outcome can teach
it. The evidence so far does **not** establish that.

`python -m src.rl.diagnose placement --episodes 200`. Two policies identical
in play rate and slot choice, differing only in where they place: one spreads
over legal tiles, one always uses `c3r12` (`run3`'s own most-used tile).

| pool | spread | concentrated | delta |
|------|--------|--------------|-------|
| baseline (frozen 4) | 34.0% | 49.0% | **+15.0** |
| punishers | 21.5% | 39.0% | **+17.5** |

The deltas are **the same**. The 2.5pp difference is far inside the ~6.9pp SE
at this n, and it points the wrong way anyway. Concentrating on one tile is
worth just as much against the punishers as against the frozen four.

So: harder, yes. Placement-discriminating, no.

### Why this probe cannot settle it

The baseline it compares against is **random spread, which is not good play**.
`c3r12` is a strong defensive tile, so the "concentrated" policy is not making
the mistake the punishers were built to catch — it is playing *one good tile
every time*, which beats scattering cards over the whole arena for reasons
that have nothing to do with concentration.

The delta therefore measures *good tile vs. random tile*, and conflates tile
quality with concentration. A real placement punisher should shrink the gap
between concentrated play and **appropriate** play, not between concentrated
play and noise.

### The tile sweep: run, and it confirms the negative

Varying the tile is the comparison the probe above cannot make, so it was run.
Win rate of the placing policy, n=150 per cell:

| tile | baseline | punishers |
|------|----------|-----------|
| *(random spread)* | 34.0% | 22.7% |
| `c3r12` in front of towers | 48.0% | 36.7% |
| `c0r15` back corner | 45.3% | 30.0% |
| `c4r15` behind the king | 33.3% | 22.0% |
| `c1r8` at the bridge | 26.7% | 23.3% |
| **range across tiles** | **21.3pp** | **14.7pp** |

Single-cell SE is 4.1pp; SE of a difference 8.2pp.

**The punishers do not discriminate more — if anything less.** Their range
across tiles is 14.7pp against the frozen four's 21.3pp. That 6.6pp gap is
itself inside the 8.2pp difference SE, so "they discriminate *less*" is not
established either. What is clear is that **the design goal — wider
discrimination — did not happen.** The two pools track each other tile for
tile, offset by a roughly constant ~11pp.

The one place they diverge is `c1r8`, the bridge: −7.3pp against the frozen
four but +0.6pp against the punishers. The punishers are *more forgiving* of
that tile, the opposite of the intent. `MIN_THREAT_TO_ANSWER` is the likely
cause — declining to answer small threats means a card dribbled at the bridge
goes unpunished — which would make that heuristic actively counterproductive
for this pool's purpose even though the ablation showed it helps the win rate.

### The finding that matters more than the pool

**Three of the four fixed tiles beat random spread**, two of them by 11–14pp:
`c3r12` +14.0, `c0r15` +11.3, `c4r15` −0.7, `c1r8` −7.3. Concentrating on one
tile is *better play than spreading* in this simulator, almost regardless of
which tile, and that holds against both pools.

If that survives a proper check, it reframes `docs/rl-training.md` §4.
`run3`'s 72%-of-placements-on-one-tile has been read as a pathology — a
symptom of a frozen tile head. It may instead be a **rational adaptation to
an environment that rewards concentration**, in which case there is no
placement lesson being missed, and the ~50% ceiling has to be explained some
other way.

This is one measurement at n=150 over four hand-picked tiles, run against
scripted bots rather than a trained policy. It is a lead, not a result. But it
is a cheap one to chase and it bears directly on the standing hypothesis.

### What would still settle the original question

1. **A responsive baseline.** Compare concentrated play against a scripted
   policy that places *appropriately* (defends where the threat is), not
   against random. That is the comparison the hypothesis is actually about.
3. **The real test:** train against `PUNISHER_POOL` and measure the argmax
   tile distribution and logit margin (`diagnose tiles`, §5.2 of the training
   doc). If the pool teaches placement, placement should spread relative to
   `run3`'s 8 tiles / 72%-on-one. Win rate alone will not answer it, and a
   lower headline against a harder pool proves nothing either way.

Until at least (1) is done, the honest summary is: **a harder opponent pool
whose teaching value is unmeasured.**

---

## 5. The bug that mattered

**Rank spell targets by elixir, never by body count.**

The first version gated on `MIN_SPELL_HITS = 3` bodies. A swarm card puts three
bodies on one tile for one payment, so *every single Goblin placement* looked
like the biggest clump on the board, and the bot answered a 2-elixir card with
a 4-elixir Fireball, over and over.

Measured, n=200 each, random policy — **higher means a weaker opponent**:

| variant | random wins |
|---------|-------------|
| `control` (frozen) | 33.0% |
| `controlplus` as first built | **44.0%** |
| ...with the defence threshold removed | 46.5% |
| ...with spells disabled | **33.0%** |
| ...with lane picking removed | 42.5% |
| `control` + lane picking only | 32.0% |

Disabling spells restored frozen `Control`'s number exactly. The spell logic
was the entire regression — the defence threshold and the lane picking were
fine, and removing the threshold made things slightly *worse*, not better.

The fix is `roles.unit_elixir_value`: a body is worth `cost / count`, so three
Goblins are worth two elixir between them rather than six. The spell then fires
only on a positive trade (`SPELL_TRADE_MARGIN`).

`tests/test_sim_punisher.py::test_a_swarm_card_is_not_worth_a_fireball` is the
regression guard. Do not relax it.

**The general lesson**, worth carrying to the next bot: an ablation against a
*random* policy is cheap (a few minutes at n=200) and it caught a change that
made the opponent 11pp worse while looking, in the code, like a clear
improvement. Run one before believing any new opponent behaviour helps.

---

## 6. Structured decks

`ArchetypeDeckSpread` samples a role structure instead of 8 uniform cards: one
tank, two spells, a building, a mini tank, a swarm, an air-defense troop if
nothing drawn already shoots air, then filler.

It is **opponent-only and opt-in** (`--structured-decks`,
`SimEnv.opponent_decks`). Switching the global sampler would change the decks
the frozen four are handed and invalidate their recorded numbers.

Two things worth knowing before turning it on:

- **It narrows variety.** With today's 12-card pool exactly one card clears
  `TANK_HP_MIN` (Giant) and exactly two are spells (Arrows, Fireball), so three
  of eight slots are constant in every structured deck. The remaining five
  still vary. This stops being true as soon as the detector learns a second
  heavy card.
- **They measured WORSE for every bot tried, including the two that cast
  spells.** Two of eight cards become close to dead weight: `affordable()`
  excludes spells, so a spell in hand is only playable when `cast_spell`
  finds a clump worth the elixir, and against a spread policy that is rare.

  | bot | normal decks | structured | cost |
  |-----|-------------|-----------|------|
  | `control` (frozen, casts nothing) | 33.0% | 44.0% | −11.0pp |
  | `controlplus` | 29.5% | 31.5% | −2.0pp |
  | `punisher` | 18.0% | 24.5% | −6.5pp |

  (Higher = weaker, so a positive move right is a loss.) Casting spells cuts
  the damage from 11pp to 2–6.5pp but does not erase it. **This is why the
  flag defaults off.** It may still be the right call against a *trained*
  policy, which clumps far more than a random one and so offers the clumps
  that make two spells worth carrying — that is untested.

---

## 7. The tank rule

`roles.heaviest` — the heaviest **troop** in hand leads the push. Relative, not
absolute.

An absolute rule ("wait for Giant") deadlocks: a deck is 8 of 12 cards and a
hand is 4 of those 8, so a bot holding out for one specific card spends much of
the match unable to spend at all. Even with structured decks guaranteeing a
Giant in the *deck*, it is in *hand* only about half the time. Leading with
Knight when Giant is not in hand is what a human does.

Buildings are excluded even when they out-HP the troops: a building does not
advance, so it cannot lead anything.

---

## 8. State, and what is not done

**Done:** both bots, the pool split, structured decks, both CLIs
(`--opponent baseline|punishers`, `--structured-decks`), the placement probe
(`src/rl/diagnose.py`), 42 tests.

**Established:** they are harder — `punisher` 18.0% against the hardest frozen
bot's 27.0%, many SE apart. The frozen four are untouched and still reproduce
31.0%.

**Measured and NEGATIVE:** they do not make placement matter more (§4). The
range across fixed tiles is 14.7pp against them versus 21.3pp against the
frozen four - the design goal did not happen. That was the point of the
branch, so treat the pool as a harder yardstick, not as the answer to the
training doc's §4.

**Not done, in the order I would do it:**

0. **Check whether concentration is simply correct here** (§4). Three of four
   fixed tiles beat random spread, two by 11-14pp. If the sim rewards playing
   one tile, `run3` concentrating is rational rather than broken and the
   premise of the training doc's §4 needs revisiting. This is the highest-value
   item on the list and it is cheap.
1. **Train something against this pool** — only if (0) does not dissolve the
   question. Nothing has been trained against it. The numbers here are all
   *random-policy* baselines, which say how hard the bots are but nothing
   about whether they teach placement. Needs a 2M-step run.
2. **The paired evaluation** from `docs/rl-training.md` §5.1 — still the
   cheapest useful work in the project, and a prerequisite for trusting any
   comparison between a run trained on this pool and one trained on the frozen
   four.
3. **Measure whether placement got more varied**, not just whether the win rate
   moved. §5.2 of the training doc is explicit that entropy is the wrong
   instrument: measure the deterministic argmax tile distribution and the
   top1-minus-median logit margin. If the punishers work, placement should
   spread; if the win rate drops and placement stays concentrated, they are
   just harder rather than more instructive.
4. **Reward shaping** (`docs/ideas/placement-shaping.md`) — the other half of
   the idea, untouched here.

One caution on (1): these bots are harder than the frozen four, so a policy
trained against them will post a *lower* headline. That is not a regression,
and comparing it to the 51% in §1 is the mistake this pool split exists to
prevent.
