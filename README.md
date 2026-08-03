# clashBot

A Clash Royale automation backend that exposes a clean
observation/action/reward API on top of a Roboflow vision pipeline.
Plug in any policy (random, scripted, RL, imitation) and let the
backend handle capture, detection, elixir bookkeeping, mouse control,
and match lifecycle.

## What's in the box

**Vision** — pixels in:

| Module                                             | Role                                                       |
| -------------------------------------------------- | ---------------------------------------------------------- |
| [`src/vision/capture.py`](src/vision/capture.py)   | `mss` + `pywinctl` window capture wrapper                  |
| [`src/vision/lifecycle.py`](src/vision/lifecycle.py) | Menu / in-match / postmatch detection + auto-rematch     |
| [`src/vision/ocr.py`](src/vision/ocr.py)           | Tower-HP reader (EasyOCR) and match-timer reader (Tesseract) |
| [`src/vision/hud.py`](src/vision/hud.py)           | Shared HUD primitives: fractional crops + glyph isolation   |
| [`src/vision/elixir.py`](src/vision/elixir.py)     | Elixir count read by matching 11 reference crops (no OCR)   |

**Game model** — what the pixels mean:

| Module                                   | Role                                                      |
| ----------------------------------------- | ---------------------------------------------------------- |
| [`src/game/cards.py`](src/game/cards.py) | `Card` / `Troop` data classes + name -> elixir cost lookup |
| [`src/game/board.py`](src/game/board.py) | 4-card hand, 9x16 arena, placement rules, tensor encoding  |
| [`src/game/state.py`](src/game/state.py) | Match time, elixir, tower HP, crowns, win/loss             |

**Environment** — the policy-facing contract:

| Module                                                 | Role                                                    |
| ------------------------------------------------------- | --------------------------------------------------------- |
| [`src/env/observation.py`](src/env/observation.py)     | Builds the structured observation dict + schema hashing |
| [`src/env/actions.py`](src/env/actions.py)             | Discrete action space + mouse executor                  |
| [`src/env/environment.py`](src/env/environment.py)     | `ClashEnv`: reset / step / observe / close, reward, JSONL recording |

**Imitation learning** — learning from recorded play:

| Module                                                 | Role                                                  |
| ------------------------------------------------------- | ------------------------------------------------------- |
| [`src/imitation/recorder.py`](src/imitation/recorder.py) | Records human play (pynput drag watcher + passive `observe()`) |
| [`src/imitation/dataset.py`](src/imitation/dataset.py) | Loads and pairs two-stream demos, with no-op downsampling |
| [`src/imitation/model.py`](src/imitation/model.py)     | Factored-head network (play / slot / tile) + checkpoint I/O |
| [`src/imitation/train.py`](src/imitation/train.py)     | Trains the factored-head policy                       |
| [`src/imitation/policy.py`](src/imitation/policy.py)   | Runs a checkpoint with inference-time action masking  |

**Entry points and tooling:**

| Module                                     | Role                                                           |
| ------------------------------------------- | ---------------------------------------------------------------- |
| [`src/main.py`](src/main.py)               | CLI driver: policy selection, `--debug`, `--record`, `--record-human` |
| [`src/calibrate.py`](src/calibrate.py)     | Interactive per-machine calibration wizard (5 phases)          |
| [`src/debug/overlay.py`](src/debug/overlay.py) | Debug overlay rendering only                               |

## Setup

1. Install [BlueStacks](https://www.bluestacks.com/) and Clash Royale.
2. Create a Python 3.10+ virtualenv and install deps:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and
   make sure it's on your `PATH` (or set `TESSDATA_PREFIX`). Only the language
   data comes from this install — `tesserocr` bundles the library itself, and
   on Windows it must come from the prebuilt wheel pinned in
   `requirements.txt` (pick the `cpXXX` matching your Python).

   Tesseract reads the **match timer**. The **tower HP** numbers are read by
   EasyOCR, which downloads ~100 MB of models to `~/.EasyOCR` on first use and
   runs on the GPU. Both recognisers are hard requirements: a missing install
   raises rather than degrading silently. See the header of
   [`src/vision/ocr.py`](src/vision/ocr.py) for why the two differ.
4. Copy `.env.example` to `.env` and set your Roboflow `API_KEY`.

## Run

```bash
python -m src.main                        # 1 episode, random policy, no UI
python -m src.main --debug                # show OpenCV overlay window
python -m src.main --record logs/run.jsonl --episodes 50
```

The first run downloads/loads the `troop-counter/8` model; subsequent
runs are fast. Override it with `--model`.

## Calibration

Several constants are inherently per-machine. Search for ``CALIBRATE``
in the codebase; the hot spots are:

1. **Window crop** (`viewport` phase) in
   [`src/vision/capture.py`](src/vision/capture.py)
   (`WINDOW_CROP_TOP/LEFT/RIGHT/BOTTOM`).
2. **Hand card pixel positions** (`hand` phase) in
   [`src/env/actions.py`](src/env/actions.py) (`HAND_CARD_POSITIONS`).
3. **Tower HP regions** (`towers` phase) in
   [`src/vision/ocr.py`](src/vision/ocr.py) (`TOWER_HP_REGIONS`).
4. **Match timer region** (`timer` phase) in
   [`src/vision/ocr.py`](src/vision/ocr.py)
   (`MATCH_TIMER_REGION`) — the `m:ss` countdown, used to anchor the match
   clock. Wrong values here are fatal: the env raises rather than run on a
   simulated clock that is ~5s fast.
5. **Elixir count region** (`elixir` phase) in
   [`src/vision/elixir.py`](src/vision/elixir.py) (`ELIXIR_DIGIT_REGION`),
   plus the 11 reference crops the phase writes to
   `src/assets/templates/elixir/`. The count is *classified* against those
   crops rather than OCR'd — it is an 11-way choice at a fixed position, so
   matching is exact and cannot misread. Re-running this phase restamps the
   region; the reader refuses templates whose stamp disagrees with the
   committed constant instead of silently reading nothing.
6. **Lifecycle pixel samples and template images** in
   [`src/vision/lifecycle.py`](src/vision/lifecycle.py) and
   [`src/assets/templates/`](src/assets/templates/). These have **no**
   wizard phase — tune them by hand.

`python -m src.main --calibrate` walks the five wizard phases (1-5) and
prints the constants to paste in; the lifecycle samples are not covered.

To redo just one constant, name its phase — `viewport`, `hand`, `towers`,
`timer`, or `elixir` — and only that phase runs and only its constant is
printed:

```bash
python -m src.main --calibrate timer      # just MATCH_TIMER_REGION
python -m src.main --calibrate hand       # just HAND_CARD_POSITIONS
python -m src.main --calibrate elixir     # region + the 11 reference crops
```

An unknown phase name is an error listing the valid ones. The `towers`,
`timer` and `elixir` phases box in-match HUD elements, so run them with a
live match on screen. The `elixir` phase additionally needs you to play a
while: it wants one labelled crop for each value 0-10, taken by keypress as
the counter passes through them. Skipping the `viewport` phase means the other phases measure
against the committed `WINDOW_CROP_*` crop (the same frame the bot sees at
runtime), so re-run `viewport` first if the window size or theme changed.

Run with `--debug` to see the captured frame and the tile grid overlay
while you tune.

## Plug in a custom policy

```python
from src.env.environment import ClashEnv
from src.env.actions import Action

def my_policy(obs):
    # obs is a dict of numpy arrays. See src/env/observation.py for shapes.
    # Return: Action(hand_index, tile_x, tile_y), or an int index, or
    # Action.no_op().
    ...

env = ClashEnv("BlueStacks App Player 1")
obs = env.reset()
done = False
while not done:
    obs, reward, done, info = env.step(my_policy(obs))
env.close()
```

## Imitation learning

Record with `--record` (bot play) or `--record-human` (your own play);
both write the format below.

```bash
# 1. Record your own play in BlueStacks (drag-style placement only)
python -m src.main --record-human demos/run.jsonl --episodes 10

# 2. Train the factored-head policy on the demos
python -m src.imitation.train demos/run.jsonl --out models/imitation.pt

# 3. Run the trained checkpoint
python -m src.main --policy imitation --weights models/imitation.pt
```

The model is **not** one 577-way softmax: a shared MLP trunk feeds
separate play / slot / tile heads, and
[`src/imitation/policy.py`](src/imitation/policy.py) masks unaffordable
slots and unplaceable tiles at inference time. Checkpoints carry the
observation `schema_hash`, so a checkpoint trained on one schema refuses
to load against another.

## Observation schema

`ObservationBuilder.build` returns:

| Key             | Shape                | Meaning                                  |
| --------------- | -------------------- | ---------------------------------------- |
| `hand`          | (4, V)               | one-hot card identity per slot           |
| `hand_costs`    | (4,)                 | elixir cost per slot                     |
| `hand_playable` | (4,)                 | 1 where elixir >= cost, 0 elsewhere      |
| `elixir`        | scalar               | 0-10, read from the HUD count             |
| `match_time`    | scalar               | seconds elapsed, timer-anchored, cap 300 |
| `time_norm`     | scalar               | match_time / 300                         |
| `phase_onehot`  | (4,)                 | normal / double / ot_d / ot_t            |
| `arena`         | (16, 9, 2*\|T\|)     | one-hot troop x color per tile           |
| `tower_hp`      | (6,)                 | normalized HP per tower                  |
| `crowns`        | (2,)                 | (friendly, enemy) crown counts           |
| `playable_mask` | (16, 9)              | 1 where friendly may place               |

`ObservationBuilder.flatten(obs)` produces a single 1-D `float32` array
for MLP-style policies.

## Recording format (JSONL, `record_format: 2`)

Observations and actions are **two independent timestamped streams**,
not one line per step. A perception cycle takes ~0.3 s — far longer than
the gap between two quick card placements — so any format that carries
one action per step silently drops or mis-attributes the extras.

```jsonc
{"type":"meta","record_format":2,"schema_hash":"…","schema":{…}}
{"type":"obs","t":1712.104,"step":0,"obs_flat":[…],"reward":0.0,
 "lifecycle_state":"in_match","lifecycle_result":null,"match_time":0.4,
 "elixir":5.0,"match_result":null,"source":"human"}
{"type":"act","t":1712.310,"action_index":416,"hand_index":2,"tile_x":4,
 "tile_y":11,"success":true,"reason":"human","source":"human"}
{"type":"act","t":1712.480,"action_index":100,"hand_index":0,"tile_x":5,
 "tile_y":11,"success":true,"reason":"human","source":"human"}
{"type":"obs","t":1713.720,"step":1,…}
```

- `obs.t` is the frame **capture** time; `act.t` is the moment the action
  was issued (bot) or the drag released (human). Same `time.time()` clock.
- No-op actions are never written — an observation with nothing attached
  *is* the no-op sample.
- Pairing happens offline in `src/imitation/dataset.py`: each action binds
  to the nearest observation captured **strictly before** it. One
  observation may take several actions (→ several training rows); ties on
  the coarse 15.6 ms Windows clock resolve backwards.
- `record_format` is checked on both append and load. Format-1 files (one
  flat line per step, action inline) are refused — their actions carry no
  timestamp, so they cannot be re-paired.

## Action space

Discrete: `1 + 4 * 9 * 16 = 577` choices. Index 0 is NO_OP; the rest
enumerate `(hand_index, tile_y, tile_x)`. Use ``index_to_action`` and
``action_to_index`` in [`src/env/actions.py`](src/env/actions.py) to convert.

## Roadmap

Items still owned by future iterations (ordered roughly by impact):

- Tighten lifecycle detection with real template assets.
- Replace OCR tower HP with a fine-tuned digit classifier. EasyOCR fixed
  the *wrong*-value problem that made Tesseract unusable here, but it still
  only reads a tower's number on roughly a third to a half of frames, and
  destruction is inferred from a long run of misses rather than observed.
  A classifier would be both faster and more consistently readable.
- Detect the "up next" card identity, not just filter it out, so the
  policy can plan ahead.
- Train a baseline policy (PPO on the flat observation, or behavior
  cloning from recorded JSONL).
