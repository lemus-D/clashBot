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

- Core loop: `ClashEnv` in `src/environment.py` — reset / step / close, reward
  shaping, JSONL recording. This is the contract everything else serves.
- Vision: Roboflow model (`troop-counter/7`) + screen capture via `mss` /
  `pywinctl` in `src/capture.py`.
- Observation: built in `src/observation.py` as a structured dict; `flatten()`
  gives a 1-D float32 array for MLP policies.
- Actions: `src/actions.py` — 577 discrete choices (NO_OP + hand x tile);
  `index_to_action` / `action_to_index` convert.
- Game model: `gameBoard.py` (9x16 arena, hand, placement), `gameState.py`
  (time, elixir, tower HP, crowns), `cardDatabase.py`, `cardClasses.py`.
- Lifecycle: `matchLifecycle.py` — menu / in-match / postmatch + auto-rematch.
- Tower HP: `towerHealth.py` via optional Tesseract OCR (degrades gracefully).
- Entry point: `src/main.py` — CLI driver with `RandomPolicy`, `--debug`, `--record`.
- Per-machine constants are marked `CALIBRATE` (capture crop, hand card pixel
  positions, tower HP regions, lifecycle samples). Keep them centralized there.

## Setup & Commands

- Python 3.10+. Create venv and install: `python -m venv .venv` →
  `.venv\Scripts\activate` → `pip install -r requirements.txt`
- Config: copy `.env.example` to `.env`, set Roboflow `API_KEY`.
- Run: `python -m src.main` (add `--debug` for overlay, `--record logs/run.jsonl
  --episodes N` to record).
- Build: none yet. Tests: none yet. (Flag if you think one is needed.)

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
  what was expected. (Optional Tesseract OCR is the one documented exception —
  it may fall back to "full" tower HP when unavailable.)

### Conciseness & Scope

- Write the simplest thing that solves the actual requirement (YAGNI) — don't
  build for hypothetical future needs.
- Solve the problem in front of you, not every edge case you can imagine. Handle
  real, known cases; fail clearly on the rest.
- Don't add config options, layers, or generality "just in case."
- Fewer moving parts is better. Reach for a new abstraction only when real
  duplication or complexity justifies it.

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
