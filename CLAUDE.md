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
- Tests: `pip install -r requirements-dev.txt` then `pytest`. Covers the
  class manifest, observation encoding and the simulator - everything
  that runs without a screen or an emulator. Build: none yet.

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
