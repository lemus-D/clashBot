# clashBot

A Clash Royale automation backend that exposes a clean
observation/action/reward API on top of a Roboflow vision pipeline.
Plug in any policy (random, scripted, RL, imitation) and let the
backend handle capture, detection, elixir bookkeeping, mouse control,
and match lifecycle.

## What's in the box

| Module                                       | Role                                                        |
| -------------------------------------------- | ----------------------------------------------------------- |
| [`src/capture.py`](src/capture.py)           | `mss` + `pywinctl` window capture wrapper                   |
| [`src/cardDatabase.py`](src/cardDatabase.py) | Static card name -> elixir cost lookup                      |
| [`src/cardClasses.py`](src/cardClasses.py)   | `Card` / `Troop` / `BlankSpace` data classes                |
| [`src/gameBoard.py`](src/gameBoard.py)       | 4-card hand, 9x16 arena, placement rules, tensor encoding   |
| [`src/gameState.py`](src/gameState.py)       | Match time, elixir, tower HP, crowns, win/loss              |
| [`src/towerHealth.py`](src/towerHealth.py)   | OCR tower HP reader (Tesseract, required)                   |
| [`src/matchLifecycle.py`](src/matchLifecycle.py) | Menu / in-match / postmatch detection + auto-rematch    |
| [`src/observation.py`](src/observation.py)   | Builds the structured observation dict                      |
| [`src/actions.py`](src/actions.py)           | Discrete action space + mouse executor                      |
| [`src/environment.py`](src/environment.py)   | `ClashEnv`: reset / step / close, reward, JSONL recording   |
| [`src/windowCap.py`](src/windowCap.py)       | Debug overlay rendering only                                |
| [`src/main.py`](src/main.py)                 | CLI driver with `RandomPolicy` and `--debug` / `--record`   |

## Setup

1. Install [BlueStacks](https://www.bluestacks.com/) and Clash Royale.
2. Create a Python 3.10+ virtualenv and install deps:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and
   make sure it's on your `PATH`. It's required for tower-HP reading; the bot
   raises if a read fails or the binary is missing.
4. Copy `.env.example` to `.env` and set your Roboflow `API_KEY`.

## Run

```bash
python -m src.main                        # 1 episode, random policy, no UI
python -m src.main --debug                # show OpenCV overlay window
python -m src.main --record logs/run.jsonl --episodes 50
```

The first run downloads/loads the `troop-counter/7` model; subsequent
runs are fast.

## Calibration

Several constants are inherently per-machine. Search for ``CALIBRATE``
in the codebase; the four hot spots are:

1. **Window crop** in [`src/capture.py`](src/capture.py)
   (`WINDOW_CROP_TOP/LEFT/RIGHT/BOTTOM`).
2. **Hand card pixel positions** in [`src/actions.py`](src/actions.py)
   (`HAND_CARD_POSITIONS`).
3. **Tower HP regions** in [`src/towerHealth.py`](src/towerHealth.py)
   (`TOWER_HP_REGIONS`).
4. **Lifecycle pixel samples and template images** in
   [`src/matchLifecycle.py`](src/matchLifecycle.py) and
   [`assets/templates/`](assets/templates/).

Run with `--debug` to see the captured frame and the tile grid overlay
while you tune.

## Plug in a custom policy

```python
from src.environment import ClashEnv
from src.actions import Action

def my_policy(obs):
    # obs is a dict of numpy arrays. See observation.py for shapes.
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

For imitation learning, record with `--record` (bot play) or
`--record-human` (your own play); both write the format below.

## Observation schema

`ObservationBuilder.build` returns:

| Key             | Shape                | Meaning                                  |
| --------------- | -------------------- | ---------------------------------------- |
| `hand`          | (4, V)               | one-hot card identity per slot           |
| `hand_costs`    | (4,)                 | elixir cost per slot                     |
| `hand_playable` | (4,)                 | 1 where elixir >= cost, 0 elsewhere      |
| `elixir`        | scalar               | 0-10                                     |
| `match_time`    | scalar               | seconds elapsed                          |
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
``action_to_index`` in [`src/actions.py`](src/actions.py) to convert.

## Roadmap

Items still owned by future iterations (ordered roughly by impact):

- Tighten lifecycle detection with real template assets.
- Replace OCR tower HP with a fine-tuned digit classifier (faster + more
  reliable than Tesseract).
- Detect the "up next" card identity, not just filter it out, so the
  policy can plan ahead.
- Train a baseline policy (PPO on the flat observation, or behavior
  cloning from recorded JSONL).
