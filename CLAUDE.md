# CLAUDE.md

Persistent context for Claude Code, loaded at the start of every session.
Keep this high-signal: things Claude would get wrong without being told.

## Project Overview

clashBot is a self-improving Clash Royale bot written in Python. Goal: a model
that plays the game and climbs the ranks via machine learning.
The backend exposes a gym-like observation / action / reward API over a Roboflow
vision pipeline, so any policy (random, scripted, RL, imitation) can plug in.
See ./README.md for the full module map, observation schema, and action space.

## Project Status & Working Style

- Early-stage. The current architecture and code quality are known to be rough
  and are actively being improved — do NOT treat existing patterns as the
  standard to preserve.
- When touching code, improving its structure is welcome and encouraged. But
  keep each change scoped to the task and explain the reasoning before large
  refactors.
- No build or test tooling exists yet. Don't assume commands that aren't here.

## Architecture at a Glance

(Current state — under active revision. Describe and improve it as it is; don't
speculatively generalize for cases that don't exist yet.)

- Core loop: `ClashEnv` in `src/env/environment.py` — reset / step / observe /
  close, reward shaping, JSONL recording. This is the contract everything else
  serves. `observe()` is the passive perceive-only cycle (used by `step` and
  the demo recorder).
- Observation: `src/env/observation.py` — structured dict; `flatten()` gives a
  1-D float32 array for MLP policies.
- Actions: `src/env/actions.py` — 577 discrete choices (NO_OP + hand x tile);
  `index_to_action` / `action_to_index` convert.
- Vision (`src/vision/`): Roboflow model (`troop-counter/8`); screen capture
  via `mss` / `pywinctl` in `capture.py`; match lifecycle (menu / in-match /
  postmatch + auto-rematch) in `lifecycle.py`; the match timer is the ONLY
  remaining OCR (`ocr.py`, Tesseract/`tesserocr`; required, raises on missing
  install, but a single unreadable frame is `None` and the caller decides).
- Tower HP is the FILL FRACTION of the on-screen HP bar (`src/vision/towers.py`),
  normalized 0.0-1.0, measured column-wise. This replaced EasyOCR on the HP
  digits (~84ms/cycle, only ~36-54% of frames legible) and deleted the learned
  max-HP denominator, whose one 5147 misread used to poison a whole match.
  Bars deplete right-to-left. Destroyed reads 0.000 — but so does a bar hidden
  behind a fight, so destruction needs a sustained run of absences (debounced
  in `state.py`, retracted if the tower reads again). Kings are excluded from
  destruction inference: a king kill is the banner's verdict, not vision's.
  Do NOT reuse the old HP-number regions for the bars — the princess number is
  drawn above the bar, and the offset differs per tower.
- Elixir is READ, not simulated: `src/vision/elixir.py` classifies the HUD
  count against 11 reference crops (`src/assets/templates/elixir/0..10.png`,
  captured by `--calibrate elixir`) rather than OCR'ing it — an 11-way choice
  at a fixed position, so matching is exact where OCR could misread. The
  display shows the FLOOR, which is the conservative direction for
  affordability. `GameState.set_elixir` makes the reading authoritative and
  demotes the simulation to carrying the fraction between reads. Templates
  are stamped with the region they were captured against; a stamp that
  disagrees with `ELIXIR_DIGIT_REGION` raises rather than reading nothing.
- Game model (`src/game/`): `board.py` (9x16 arena, hand, placement rules),
  `state.py` (time, elixir, tower HP, crowns), `cards.py`, `classes.py`.
- NAMING CONTRACT — `src/game/classes.py` owns it, and it is GENERATED, not
  hand-written. `class_manifest.json` comes from `get_model(...).class_names`
  via `python -m src.main --derive-classes`; runtime reads the manifest (no
  network at import) and `ClashEnv._load_model` calls `verify_against_model`
  to raise if it has gone stale. A hand-written list broke this twice: plural
  entries blanked Minions/Archers/Spear Goblins/Goblins out of the arena
  tensor entirely, and later eight arena-2 names the model cannot emit made
  2304 observation values permanently zero. Do not reintroduce one.
  - TWO index spaces, not one. `ARENA_CLASSES` (13) and `CARD_CLASSES` (12)
    are overlapping sets where neither contains the other: Goblin Brawler is
    an arena unit with no card because it spawns from Goblin Cage, and every
    spawn-only unit (Lava Pups, Golemites, …) is the same. `arena` is sized
    `2 * |ARENA_CLASSES|`, `hand` is `|CARD_CLASSES|`. A future spawn-only
    unit needs no special-casing — it lands in one list and not the other.
  - Order is APPEND-ONLY (`merge_preserving_order`). Class order is channel
    order; old index i must stay new index i or a trained checkpoint cannot
    be migrated by scattering its input-layer columns, and adding one arena
    class shifts the entire enemy channel half. A class that disappears from
    the model is KEPT as a dead channel rather than dropped, for the same
    reason. Never sort the manifest, and never trust the model's own order.
  - `CARD_COSTS` keys are validated against `CARD_CLASSES` at import and
    raise on mismatch; the costs themselves stay hand-written, since a
    detector knows what a card looks like and not what it costs.
  - `king tower` / `princess tower` are real model classes deliberately
    excluded via `IGNORED_ARENA_CLASSES`: tower SKINS change how towers
    look, so those two detect unreliably. Don't add them, and don't try to
    use a princess tower vanishing from detections as a destruction signal —
    same unreliability. Tower state comes from the HP bar.
- SPELL PLACEMENT: troops may only be deployed on the friendly half (plus a
  lane opened by a destroyed tower); SPELLS may be cast ANYWHERE. Which cards
  are spells is hand-maintained in `cards.py` as `SPELL_CARDS` (validated
  against `CARD_CLASSES` at import) - the detector cannot tell you, same as
  cost.
  - ONE IMPLEMENTATION: `board.placement_allowed()`, written in the PLACER's
    own frame. `GameBoard.is_placeable` is a thin wrapper;
    `Simulation.is_placeable` mirrors the enemy's tiles into that frame and
    calls the same function. Do not write a second copy - there was one, and
    the two drifted in both directions at once (see below).
  - An activated enemy king tower does NOT unlock the enemy half. The board
    used to grant that and the simulator never did, so the observation's mask
    opened the whole enemy half the moment the enemy king took ~2% chip
    damage. 60% of `run1`'s placement attempts were refused by the env, all
    on the SAME tile, because deterministic eval re-picks its top-ranked tile
    every step. Territory expands only when a princess tower is DESTROYED.
  - `RIVER_ROW` (7, in the placer's frame) never takes ground troops, even
    once its lane opens. The simulator was missing this one.
  - `tests/test_sim_engine.py::TestMaskAgreesWithEnv` compares the
    observation mask against the env tile-by-tile in three states. Keep it.
  - The observation carries the TROOP mask plus a per-slot `hand_is_spell`
    (4,) flag, not four full 144-tile masks: spell-vs-troop is the only
    card-dependent rule, so 4 floats say what 4x144 would.
  - Anything that masks tile choices MUST resolve the slot FIRST, then pick
    the mask. Masking tiles before knowing the slot forbids every legal
    spell target on the enemy half, which is exactly the bug this replaced:
    `playable_mask` was the only gate, so Fireball and Arrows could never
    reach an enemy tower and a policy would correctly have learned that two
    of its twelve cards were worthless. See `imitation/policy.py` and both
    RandomPolicy implementations for the ordering.
- Match clock: elixir is simulated but time is NOT trusted to simulation. The
  match is detected from the elixir bar, which is already up during the 3-2-1
  countdown, so `start_match()`'s stamp is ~5s early; `anchor_match_clock()`
  re-derives `match_start_time` from the first trustworthy `MatchTimerReader`
  reading (once per match) and `ClashEnv` raises if that never happens.
  `get_current_match_time()` is clamped to 300s for `time_norm`;
  `get_elapsed_seconds()` is the uncapped one for timeouts.
- Recording format (`record_format: 2`, defined in `env/environment.py`, used
  by BOTH `--record` and `--record-human`): a `{"type": "meta", ...}` header
  (record_format + observation `schema_hash` + `schema_descriptor()`), then two
  independent timestamped streams — `{"type": "obs", "t": ...}` per perception
  cycle and `{"type": "act", "t": ...}` per placement. NOT one line per step: a
  cycle is ~0.3s and a human can place several cards inside one. No-ops are not
  written. Pairing is offline (`dataset.py`): each action binds to the nearest
  observation captured strictly before it; an obs may take N actions (N rows)
  or none (a no-op row). `obs.t` is frame-capture time, `act.t` is issue /
  drag-release time. Format-1 files and cross-schema data are refused on both
  append and load; checkpoints also carry the schema hash.
- Imitation learning (`src/imitation/`): `recorder.py` records human play —
  pynput mouse watcher maps hand→arena drags to timestamped actions while
  `env.observe()` runs passively; adds `"source": "human"`.
  `dataset.py` loads and pairs demos with no-op downsampling; `model.py` is a
  factored-head net (shared MLP trunk → play/slot/tile heads, NOT one 577-way
  softmax); `train.py` trains it; `policy.py` runs a checkpoint with
  inference-time masking of unaffordable slots and unplaceable tiles.
  PyTorch; this machine has an RTX 4070 — use the cu128 CUDA build, at the
  torch version torchvision pins (see requirements.txt for the command).
- Postmatch verdict: the episode does NOT end on the first POSTMATCH frame.
  POSTMATCH is declared as soon as the OK button matches, and the OK button
  renders while the "Winner!" label and the crown rows are still animating in,
  so that frame answers neither who won nor by how much — the first human demo
  session labelled three won matches `None`/`loss`/`None` this way.
  `resolve_done` keeps perceiving for up to `POSTMATCH_SETTLE_SEC` (2s) and
  ends as soon as BOTH the outcome and the crowns are read; `step` downgrades
  placements to no-ops inside that window so the policy can't click the
  postmatch screen. The terminal reward is paid once, on the closing step,
  from `state.match_result` rather than that frame's `signals.result`.
  Win/loss is decided by the COLOUR of the matched label, not by template
  score: `victory.png` and `defeat.png` are the SAME word ("Winner!") drawn
  cyan over the friendly row and pink over the enemy one, so each scores high
  on the other's label (0.897 vs 0.750 measured, around a 0.80 threshold)
  while the colours are 45% vs 0%. Scores are only the mid-animation
  tiebreak, compared — never `if/elif` priority, which handed every double
  hit to "win". If the label never resolves, the crown score decides.
- Crown score: `src/vision/crowns.py` reads how many crowns each side won off
  the POSTMATCH screen, once per match, and `GameState.set_final_crowns`
  overwrites the mid-match inferred tally with it (that tally comes from
  tower-destruction debouncing and drifts). Crowns are counted as connected
  GOLD BLOBS, not fixed slot positions, because the winner's row is drawn
  larger. Rows do NOT move — friendly/blue is always the bottom one — so each
  region is one fixed side; boxing them in the wrong order during calibration
  transposes the score. Cushion colour (magenta=enemy, blue=friendly) is
  checked only to confirm the frame really is the postmatch screen, which is
  what keeps a crown score from being read off an in-match frame if the
  lifecycle state is wrong. Terminal reward scales with the crown margin
  (`TERMINAL_BASE_REWARD` +/- `CROWN_MARGIN_REWARD` per crown): a 3-crown win
  pays +14, a 1-crown win +10.
- Simulator (`src/sim/`): a headless Clash Royale model that produces the
  SAME observation schema as `ClashEnv`, so a policy trained in sim runs on
  BlueStacks unchanged. ~650x real time, ~11k matches/hour vs ~18 on
  hardware. This is the RL training backend; the vision pipeline is the
  deployment backend.
  - `env.py` does NOT build observations. It fills a real `GameBoard` and a
    `_SimStateAdapter` and calls the real `ObservationBuilder`. One encoder,
    two backends - duplicating it would drift immediately. That adapter is
    the seam; keep it.
  - `units.py` is GENERATED from real game data. `unit_stats.json` comes
    from `tools/derive_unit_stats.py`, which reads RoyaleAPI's cr-api-data
    (extracted from the game client) and normalises to TOURNAMENT STANDARD =
    displayed level 11. Adding a unit means editing that script and
    regenerating, not hand-editing the JSON.
    - The per-level arrays start at each card's OWN level 1, and a card's
      first level depends on rarity (a Rare's level 1 is displayed level 3).
      Index by rarity — `LEVEL_INDEX` — or you silently mix power levels.
    - This replaced a hand-written table whose numbers were recalled rather
      than transcribed. They were roughly LEVEL 1 values (Knight 660 vs a
      real 1766). The hardcoded TOWER figures were recalled too and also
      wrong, just less so (princess 2534/109 vs a real 3584/128), so units
      were ~1.9x too weak *relative to towers* and the sim was heavily
      tower-favoured. Errors also ran 1.03x–2.68x *between* cards, distorting
      which unit beats which. Anything hand-entered here must state its
      provenance.
    - `Confidence` is still per-stat (`measured`/`wiki`/`guess`) and
      `randomize()` perturbs by it. Everything is `wiki` now, which means the
      NUMBERS are real — not that the simulation built on them has been
      validated against actual play. `--no-randomize` is for evaluation only.
    - Arrows' radius comes out at 1.4 real-game tiles against a commonly
      cited 4.0; Fireball's 2.5 matches exactly, so the conversion is right
      and the Arrows field likely describes one arrow, not the volley. Left
      as the data says and flagged, not silently overridden.
  - Sim fidelity is the CEILING on everything trained here. `--stats` prints
    how much of the table is actually trusted.
  - STAGED CARDS: `unit_stats.json` models 14 cards the detector CANNOT
    emit yet - arena 2 (skeleton, valkyrie, bomber, tombstone), arena 3
    (barbarian, battleram, megaminion, cannon) and arena 4 (wizard,
    firespirit, electrospirit, skeletondragon, infernotower, bombtower).
    They carry `staged: true` and are deliberately ABSENT from the class
    manifest, so they have no observation channel and `Deck` rejects them.
    Adding them to the manifest early is the bug reverted in 309117a: every
    one becomes a permanently-zero input. When the vision model gains them,
    `--derive-classes` promotes them and no simulator work is needed.
    `validate_against_manifest` therefore checks ONE direction - every
    detector class needs stats; extra staged stats are fine.
  - Mechanics behind those cards, all read from game data rather than
    invented: `splash_radius` (Valkyrie carries it on the character as
    `area_damage_radius`, ranged units on their projectile's `radius`),
    `kamikaze` (spirits AND Battle Ram die after one hit - routed through
    `_on_death` so the Ram still becomes two Barbarians), `charge_range` /
    `charge_speed_mult` (Battle Ram; charge only multiplies SPEED, the
    impact is normal damage plus kamikaze, and it resets on impact or a
    target switch), `ramp` (Inferno Tower's three stages, which reset when
    the target changes - that reset IS the counterplay), and `death_damage`
    (Bomb Tower's bomb, modelled as an explosion rather than a spawned
    unit).
  - TARGETING follows the real game's rules, which are subtler than they
    look and were got wrong first time:
    - A troop WALKING toward a tower diverts to an enemy entering sight
      range - ordinary distraction, and it works.
    - A troop that has already STARTED HITTING a structure is committed
      (`Entity.locked_on_structure`) and ignores troops beside it. This is
      why prising a Royal Giant off your tower in the real game needs a stun
      or a displacement card, not just a distraction unit.
    - A troop target is HELD until it dies or passes `LEASH_FACTOR` x sight
      range. Units do not shop around mid-fight.
    - Building-only attackers (`aggro_range` 0) never divert at all.
    - Sight ranges are the game's own `sight_range`; building-targeters
      genuinely have longer ones, which is why they are pulled from further.
    - Range is measured between HITBOX EDGES: `attack_range + both radii`.
  - PATHING routes via a bridge only when the goal is ACROSS the river
    (`arena.same_side`). Keying it to which half the unit OWNED was a bug -
    defenders walked to the bridge instead of at an enemy standing next to
    them, which is what looked wrong in the viewer.
  - DECKS are sampled per episode from `CARD_CLASSES` (`DeckSpread`), both
    sides independently. A policy trained on one fixed eight learns that
    eight, not the game. This is survivable only because hand identity is IN
    the observation, so the policy can see what it holds. `min_troops` stops
    an all-spell sample. The pool grows on its own when the vision model
    gains cards. Evaluate with `--no-decks` so runs stay comparable.
  - LEVELS vary per episode (`LevelSpread`). Ladder play does not match
    players exactly, so each side samples a card level around tournament
    standard and its towers sample again around that - card level and king
    level progress together but not in lockstep. `unit_stats.json` carries
    HP/damage tables for displayed levels 9-14; only HP and damage scale in
    Clash Royale, not speed/range/hit speed.
    - LEVEL IS HIDDEN STATE and must stay that way. The detector reports
      "knight" with no level, so it is absent from the observation by
      construction and the policy has to be robust to not knowing rather
      than condition on it. `tower_hp` is a normalised FILL FRACTION, which
      is what keeps a level-14 tower from being distinguishable from a
      level-9 one at full health. Do not add absolute HP to the observation.
      Sampled levels appear in `step()`'s `info["levels"]` for debugging
      only.
    - Level variation widens the outcome distribution, so it adds variance
      to any benchmark. TRAIN with it on, EVALUATE with `--no-levels` so
      runs stay comparable.
  - `ObservationNoise` exists because the sim sees perfectly and the vision
    pipeline does not: detection dropout, position jitter, phantom units,
    and stale tower bars (an occluded bar holds its previous value, exactly
    as `GameState` does). Defaults are GUESSES - they should be replaced
    with rates measured off recorded detections.
  - Step cadence is 0.25s (4 Hz), MEASURED: `ClashEnv.step_period_sec` is
    0.25 and real recordings hold it (median 0.251s, p10 0.250 over 2098
    in-match cycles; p90 0.523 is a missed slot). Do not change it
    independently of the real loop - matched timing is the whole point.
  - Engine ticks at 20 Hz (`TICK_DT`), five sub-steps per observation. At
    4 Hz a fast unit would skip past its own attack range between frames.
  - Units, buildings and towers are all one `Entity` type; towers are
    immobile entities with a `tower_key` matching `TOWER_KEYS`.
  - Reward adds to the real env's shaping: explicit `TOWER_DESTROYED_REWARD`
    / `TOWER_LOST_PENALTY` on top of the HP swing, and a small per-step
    `ELIXIR_CAP_PENALTY` for sitting at 10 (wasted regeneration).
  - Opponents (`opponents.py`) are callables taking an `OpponentView` - a
    deliberately narrow view that CANNOT see the policy's hand or elixir,
    because a benchmark that can cheat is not a benchmark. They reason in
    their OWN frame and are mirrored at placement, so there is one
    coordinate convention. Three scripted ones exist: `bigspender` (dumps
    the priciest affordable card above 5 elixir), `cycler` (cheapest card
    as often as possible), `tankandsupport` (saves for a tank at the bridge,
    then a cheap unit BEHIND it). Plus `idle`, which plays nothing.
    - FREEZE these. They are the only stable yardstick: self-play win rate
      sits at ~50% by construction and measures nothing, so never tune a
      scripted bot to beat the current policy.
    - They DEFEND FIRST (`ScriptedOpponent._defend`): an enemy inside their
      half is answered by the cheapest affordable non-building-targeter,
      placed one row in front of the DEEPEST invader. Before this none of
      them defended at all and a random policy beat them ~47% by walking
      cards into an empty lane.
    - Each has a `reaction_s` delay measured from FIRST SIGHTING of an
      invader, and keeps attacking during it. A frame-perfect defender is
      not a harder opponent, it is an unrealistic one, and a policy trained
      against it learns to beat something that is not on ladder. The timer
      runs until the half is clear, so reinforcements cannot re-delay a
      response indefinitely.
    - Placement carries role: defence intercepts the threat, a beatdown tank
      goes deep (`BACKLINE_ROW`) so the push gathers behind it, chip goes to
      the bridge (`PUSH_ROW`).
    - Four archetypes, driven by a `STYLES` table rather than bespoke code:
      `cycle`, `control`, `beatdown` and `dump` (the last is a stress test,
      not a real archetype). A style sets reaction delay, defensive reserve
      and where offence lands, so a new archetype is a table entry.
    - `control` is the only one that converts a won defence into an attack:
      once its half is clear, the defenders that SURVIVED are already paid
      for, so it adds support behind the most advanced one. It also holds a
      `reserve` and chips only above 9 elixir.
    - RandomPolicy baseline, 100 episodes PER OPPONENT, sampled decks +
      noise + levels (the training condition): bigspender 35% /
      control 36% / cycler 27% / tankandsupport 27%, 31% overall.
      Measured 2026-08-25, post placement-mask fix, via
      `python -m src.sim.run --episodes 400 --opponent <the four>`.
      Supersedes an n=80 pre-fix baseline that read 24% overall - it was
      both stale AND undersampled.
    - Control concedes the FEWEST crowns (1.27 vs 1.49-1.64) but does not
      win most - it defends well and closes badly, because the reserve and
      the 9-elixir chip gate make it passive. Honest characterisation, not a
      bug; tune it only with a re-baseline.
    - `bigspender` stays the EASIEST rung for a TRAINED policy (run2 57%):
      dumping elixir the moment it passes 5 starves its own defence. That is
      the archetype behaving correctly. Against the RANDOM policy it and
      `control` tie at ~35% - punishing a dumped push takes a policy that
      answers it, so this rung only separates once something is learning.
    - Re-baseline after ANY sim-fidelity change — the yardstick is the
      opponents' BEHAVIOUR, which is frozen, not the numbers it produces.
      This has moved a lot as fidelity improved: 2-17% with the broken stat
      table, 18-30% once stats were real, 36-47% once defenders stopped
      walking to the bridge, 31-49% once they actually defended.
    - Self-play and a frozen-checkpoint league are still to come; sample the
      pool per episode rather than graduating through it, or the policy
      forgets how to beat the simple ones.
  - Scripted opponents do not use spells. That is a simplicity choice, not
    a limitation - aiming one well needs to know where the enemy has
    clumped, which is more judgement than a fixed yardstick should have.
    `OpponentView.affordable(exclude_spells=False)` opts in.
  - `python -m src.sim.run --watch` is the visual debugger: TRUTH on the
    left, OBSERVED (post-noise, what the policy actually gets) on the right.
    Noise events are RECORDED by `env.py`, never inferred by comparing the
    two panels - jitter moves a unit to a neighbouring tile and would be
    mislabelled a phantom.
    - `--policy models/run1.pt` watches a trained checkpoint instead of the
      random baseline (deterministic, matching the benchmark; `--sample` for
      the stochastic policy). `--opponent` takes a name, a comma list, or
      `all` and round-robins episodes through them.
    - A third REWARD panel appears whenever a `RewardTrace` is passed:
      running return, per-source cumulative split, the % of steps paying
      nothing, and a per-step sparkline. Recent placements are ringed on the
      TRUTH panel, refused ones crossed. This is what found the mask bug -
      the per-source split made a constant `invalid_action` drip obvious
      where a single scalar reward had hidden it.
- RL training (`src/rl/`): PPO against the scripted opponent pool.
  `python -m src.rl.train --total-steps 2000000 --out models/ppo.pt`,
  `--smoke` for a wiring check, `--eval-only CKPT` to benchmark. ~660
  steps/sec on the 4070, so 2M steps is under an hour.
  - `policy.py` is an actor-critic over the FACTORED action (play / slot /
    tile) with a value head. MASKING is the part to be careful with:
    - the tile mask DEPENDS ON THE SLOT (a spell may be aimed anywhere, a
      troop may not), so the slot is sampled FIRST and the mask built from
      it. Anything that masks tiles before knowing the slot re-creates the
      dead-spell bug.
    - masks are STORED with the rollout, not rebuilt at update time; two
      implementations of the same rule would drift.
    - masked logits use a large FINITE negative, not `-inf`, which NaNs
      through a softmax if a row is fully masked.
    - entropy sums the component entropies UNWEIGHTED. Weighting slot/tile
      by P(play) is more correct and was measurably wrong: the policy plays
      on ~4% of steps, so the placement heads got 4% of the exploration
      bonus - effectively `ent_coef` 0.0004 against a 144-way choice.
      Entropy went from 0.2 to 5.7 (max ~7.05) on the fix.
  - `vec_env.py` batches SimEnvs. Synchronous - it buys one batched forward
    pass, not parallel simulation. Envs AUTO-RESET, and each gets its OWN
    opponent instance (a shared bot would have one committed lane driven by
    N matches). GAE must not bootstrap through a reset; there is a test.
  - `evaluate.py` is the benchmark: fixed `EVAL_SEED` distinct from training
    seeds, same distribution as training by default (`clean=True` pins
    sampling off). Batched - one env at a time made evaluation dominate
    training wall-clock. The headline number EXCLUDES `idle`, which any
    working policy beats.
  - Checkpoints carry the observation `schema_hash` and refuse to load
    across a change. The schema has moved four times; a silent load would
    read the wrong channels while appearing to work.
  - TWO ARCHITECTURES, `--arch {mlp,conv}`, both in `policy.py` behind a
    shared `FactoredPolicy` base that owns ALL the masking. Default is
    `conv`. A checkpoint records which one built it; one with no `arch`
    field predates the split and is an MLP (run1 and run2 both load).
    - `mlp` is the original flat baseline. Its `tile_head` is
      `Linear(256, 144)`: 144 independent weight vectors with nothing
      connecting tile 37 to tile 38, so a lesson learned at one tile
      teaches its neighbour nothing. Measured, its tile head never left
      ~97% of the entropy its mask allows, across two full runs.
    - `conv` runs a 3x3 stack over the 9x16 arena and makes the tile head a
      1x1 conv, so all 144 logits come from ONE shared kernel and every
      placement trains it. 137k parameters against the MLP's 4.36M - the
      MLP's tile head alone is bigger than the whole conv net.
    - NO SCHEMA CHANGE. Everything the conv stack reads is already in the
      observation or is a constant, so `schema_hash` is unmoved and no
      recording is invalidated. The arena is recovered from the flat vector
      via `observation.FIELD_OFFSETS`, never a hardcoded offset.
    - 33 CHANNELS = 26 arena one-hots + `playable_mask` + 2 coordinate + 2
      tower footprint + 2 tower HP. The last five are the whole design:
      - COORDINATE CHANNELS ARE LOAD-BEARING, not a refinement. A conv is
        translation equivariant and this game is not: row 15 is your king's
        pocket, row 8 is the bridge. Without them the shared kernel cannot
        tell those apart and `conv` is strictly WORSE than `mlp`. Same
        reason for the learned `tile_bias` (144 params added to the
        logits) - it restores the MLP's one real strength, memorising that
        a specific tile is special.
      - TOWER CHANNELS exist because the towers are NOT in `ARENA_CLASSES`
        (skins make them undetectable, see `IGNORED_ARENA_CLASSES`), so
        nothing in the arena one-hots ever marks where a tower is. HP is
        painted at the tower's own tile, which turns "index 0 of a 6-vector"
        into a LOCAL feature and lets "defend the damaged side" be one
        pattern learned once for both lanes.
      - The FOOTPRINT channel is separate from HP because a destroyed tower
        reads 0 and so does every empty tile. HP alone cannot distinguish
        "tower dead here" from "no tower here", and late-match play turns
        on exactly that.
      - `playable_mask` was already a 16x9 map being flattened away. It
        carries the river, the friendly half and any lane opened by a
        destroyed tower - the fixed geometry, for free.
    - `board.TOWER_TILES` holds the tower tiles because a policy cannot
      import the simulator; `sim/arena.py` keeps the continuous positions
      for combat and ASSERTS the two agree, so they cannot drift. Binning
      is `int()`, the same truncation `sim/env.py` uses for units - a second
      rounding rule is how the placement predicate drifted. Consequence:
      the kings are at x=4.5 and read one tile left of centre. Known,
      accepted for consistency, not a bug to fix twice.
    - The spatial half BYPASSES `RunningNorm` - one-hots, a boolean mask,
      coordinates and fill fractions are all already 0..1, and normalising
      a sparse one-hot by its own tiny variance amplifies noise. Only the
      75-float non-spatial slice is normalised (`_VectorNorm`), which still
      accepts the FULL observation so `net.norm.update(obs)` is unchanged.
  - ENTROPY IS LOGGED PER HEAD (`entropy_play` / `_slot` / `_tile`), not
    just summed. The sum hid the single most important fact about both
    completed runs: only ONE of the three heads ever learned. play collapses
    to ~1% of its maximum while slot sits at ~96% and tile at ~97% of what
    its mask allows. A summed 5.7 of 7.05 looks healthy and is not.
    - The tile ceiling is `ln(~87)`, NOT `ln(144)` - the mask only offers
      about 87 legal tiles - so normalise against the mask or the head looks
      less frozen than it is.
    - The SLOT head being equally frozen is the standing argument against
      the conv being sufficient: it is a 4-way choice with no spatial
      structure at all, so if the real problem is that neither placement
      head receives a usable gradient, a conv improves sample efficiency of
      a signal that is not there. If tile entropy stays pinned under `conv`,
      that is the refutation, and `docs/ideas/placement-shaping.md` (which
      manufactures the missing signal) becomes the answer instead.
  - REWARD SHAPING was the thing that unblocked learning, and it was found by
    MEASURING the reward distribution rather than reasoning about it. Before
    the trade reward: 91% of steps produced exactly zero reward and 52% of an
    episode's whole signal was the win/loss bit after ~700 decisions. After:
    78% silent, terminal down to 25%. `info["reward_parts"]` exists so this
    can be re-audited; the single scalar is what hid it.
  - Three hypotheses were tested and DISPROVED before that. Do not re-try
    them without new evidence:
    - "ent_coef too high" - the entropy GRADIENT measures 0.000, because at
      maximum entropy it vanishes. Flat entropy is a SYMPTOM of a frozen
      policy, not a cause.
    - "rollout window too short" - quadrupling `rollout_steps` changed
      nothing; the credit horizon is set by lambda, not the window.
    - "credit horizon too short" - sweeping `gae_lambda` 0.95 -> 0.999
      (19 -> 250 steps) moved KL around noisily and win rate not at all.
    The one measurement that DID pay was gradient norms per loss term, which
    found the critic driving the shared trunk 23:1.
  - RESULTS. Quote these ONLY at 100 episodes per opponent (400 scored
    games), full randomization, which is the training condition:

        policy        steps   overall  bigspndr  control  cycler  tank+sup
        random          -       31%      35        36       27      27
        run1          819k      44%      49        45       43      41
        run2          2.0M      50%      57        52       49      41
        run2.best     1.64M     50%      49        57       49      45

    Learning is real and large: run2 is +19pp over random, ~5 sigma at
    n=400. Darwin's bar is 60-70%.
  - EVERY WIN RATE PREVIOUSLY RECORDED HERE WAS A 16-GAME READ AND WAS
    WRONG (the diagnostics - entropy, ev, KL, play_rate - were fine).
    `run1` reads 35% at n=16 and 44% at n=100; the "20 -> 27 -> 31 -> 34 ->
    33, FLAT from update ~120" curve and run2's apparent 25% -> 48% jump
    were both sampling noise. The old "plateau" was largely a measurement
    artifact, not a ceiling. Benchmark with
    `--eval-only CKPT --eval-episodes 100`; the in-training `--eval-every`
    passes default to 16 and are a progress indicator, NOT a result.
  - `run2` is the first run ever taken to COMPLETION: 488/488 updates, 2M
    steps, ~700 sps, ~50 min on the 4070. Same hyperparameters as `run1`
    (gamma 0.999, lambda 0.99, seed 7) so the placement-mask fix was the
    only difference. Critic ev 0.82-0.94, KL ~0.000 by the end (LR annealed
    to 6e-7), play_rate ~0.042, no draw-rate blowup.
    - Training past run1's kill point is worth +5pp (44 -> 50), but that is
      only ~1.5 sigma at n=400. Suggestive, not established.
    - `run2` and `run2.best` (update 400) TIE at 50%, so the last 88
      updates and the tail of the LR anneal bought nothing.
  - The placement-mask fix shows NO training effect. Two independent
    measurements now: re-evaluating `run1` under corrected rules moved the
    headline 33% -> 35% at n=16 (refusals 60% -> 0%), and `run2` tracks
    `run1` across run1's whole range. The mask was a real bug and worth
    fixing; it was not the thing holding the win rate down.
  - OPEN QUESTION, and still the most likely ceiling: entropy moved only
    5.72 -> 5.64 (max 7.05) across the FULL 2M steps, so the tile head is
    near-uniform over 144 tiles. The policy gained 19pp over random without
    ever learning WHERE to place - it learned WHEN. A 3963-float MLP
    predicting 144 independent tile logits is a poor fit for a spatial
    problem; a conv over the 9x16 grid would let it generalise "near my
    tower" instead of learning 144 unrelated numbers. Try that before more
    hyperparameter work.
  - `tankandsupport` is 41% for BOTH runs - the only opponent that did not
    move across 1.2M extra steps, and now the clearest single target. Note
    it is also the hardest for the random policy (27%), so the archetype is
    genuinely the top rung, not a quirk of one checkpoint.
  - `docs/ideas/placement-shaping.md` holds the unbuilt ideas for the same
    problem: a potential-based spatial reward gradient, and scripted
    opponents that punish bad placement. Thoughts, not plans.
  - Per-game numbers vary a LOT at n=16: the same checkpoint scored cycler
    at 19% and 50% on two different seed sets. Do not read a 16-game
    per-opponent number as a result.
  - `play_rate` is logged every update. ~0.026 is the sustainable rate given
    elixir regen, so a value far below that is no-op collapse and a value
    far above it means the affordability mask is broken.
- Entry point: `src/main.py` — CLI driver: `RandomPolicy`, `--debug`,
  `--record`, `--record-human`, `--calibrate`.
- Debug overlay: `src/debug/overlay.py`. Calibration wizard: `src/calibrate.py`.
- Per-machine constants are marked `CALIBRATE` (capture crop, hand card pixel
  positions, tower HP-bar regions, match timer region, elixir digit region,
  postmatch crown rows, lifecycle samples). Keep them centralized there.
- `src/calibrate.py` has six phases, selectable individually by name
  (`viewport` / `hand` / `towers` / `timer` / `crowns` / `elixir`); bare
  `--calibrate` runs all six. `crowns` is the one phase wanting the POSTMATCH
  screen rather than a live match. Phases 2-6 report fractions of the CROPPED viewport, so when
  `viewport` is skipped the frame comes from `ScreenCapture` with the committed
  `WINDOW_CROP_*` — never from the raw window grab, or every fraction is wrong.
  `elixir` is the only phase that also WRITES files (the reference crops), and
  its live preview must not sit over the game viewport: `grab()` re-reads the
  window rect every call, so an overlapping window gets captured instead.

## Setup & Commands

- Python 3.10+. Create venv and install: `python -m venv .venv` →
  `.venv\Scripts\activate` → `pip install -r requirements.txt`
- Config: copy `.env.example` to `.env`, set Roboflow `API_KEY`.
- Run: `python -m src.main` (add `--debug` for overlay, `--record logs/run.jsonl
  --episodes N` to record).
- Calibrate: `python -m src.main --calibrate` for all four phases, or
  `--calibrate <viewport|hand|towers|timer>` for one (`towers` / `timer` need a
  live match on screen).
- Record human demos: `python -m src.main --record-human demos/run.jsonl
  --episodes N` — human plays in BlueStacks (drag-style placement only).
- Train imitation policy: `python -m src.imitation.train demos/run.jsonl
  --out models/imitation.pt`; run it:
  `python -m src.main --policy imitation --weights models/imitation.pt`.
- Regenerate the class manifest: `python -m src.main --derive-classes`
  (needs API_KEY; announces any schema change it causes).
- Train a policy: `python -m src.rl.train --total-steps 2000000
  --out models/ppo.pt` (add `--smoke` for a fast wiring check,
  `--arch mlp` for the flat baseline instead of the default conv).
  Benchmark one: `python -m src.rl.train --eval-only models/ppo.pt`.
- Tests: `pip install -r requirements-dev.txt` then `pytest`. Covers the
  class manifest, observation encoding, the simulator and the RL stack -
  everything that runs without a screen or an emulator. Build: none yet.

## Coding Practices

- Match Python 3.10+ idioms; use type hints on public functions.
- Naming: PascalCase for classes; snake_case for functions and variables.
- Keep the `ClashEnv` API and the observation / action schemas stable unless
  deliberately changing them — recorded JSONL and policies depend on them. Call
  out any schema or shape change explicitly.
- Separate concerns: capture/vision, game-state modeling, env/reward, and policy
  should stay decoupled; don't let them bleed into each other.
- Fail loud, not silent: raise specific, descriptive exceptions rather than
  swallowing errors or quietly degrading. Messages should say what failed and
  what was expected. Tesseract (the match timer) and the elixir reference
  crops are hard requirements, not an exception to this rule.

### Conciseness & Scope

- Write the simplest thing that solves the actual requirement (YAGNI) — don't
  build for hypothetical future needs.
- Solve the problem in front of you, not every edge case you can imagine. Handle
  real, known cases; fail clearly on the rest.
- Don't add config options, layers, or generality "just in case."
- Fewer moving parts is better. Reach for a new abstraction only when real
  duplication or complexity justifies it.
- When responding in terminal, brevity and conciseness is important if followups are needed for further clarification they will be asked

## Machine Learning Notes

- Make runs reproducible: seed RNGs and log the config used for a run.
- Observation and reward changes ripple into recorded data and trained policies
  — note when shapes or semantics change so old runs/policies aren't silently
  misread.

## Git & Commits

- One logical change per commit; don't bundle unrelated edits.
- Never commit `.env` / secrets, `logs/` recordings, downloaded model files, or
  the `.venv`.

## Always / Never

- ALWAYS keep the env API and observation/action schemas consistent unless the
  task is explicitly to change them — and announce schema changes.
- ALWAYS ask before adding a new dependency.
- Prefer improving the rough existing code over preserving it, but keep changes
  scoped and explain them first.
- When unsure between two designs, lay out both and let me choose.
- NEVER hardcode per-machine pixel values outside the marked `CALIBRATE` spots.
- NEVER commit secrets, recordings, model files, or the venv.
