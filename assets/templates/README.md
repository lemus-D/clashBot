# Lifecycle template images

Place small PNG crops of these UI elements here so `MatchLifecycle` can
detect game start/end with high confidence. The filenames must match
the keys in `TEMPLATE_FILES` in [`../../src/matchLifecycle.py`](../../src/matchLifecycle.py).

| File                | What to capture                                                  |
| ------------------- | ---------------------------------------------------------------- |
| `battle_button.png` | The "Battle" button on the main menu                             |
| `ok_button.png`     | The "OK" button shown on the post-match summary                  |
| `victory.png`       | The yellow/gold "Victory" banner shown after winning             |
| `defeat.png`        | The blue "Defeat" banner shown after losing                      |

How to capture:

1. Run the game, open the screen showing the element.
2. Take a screenshot, crop tightly around the element (a few extra px of
   padding is fine), save as PNG into this folder with the exact name
   above.
3. Re-run the bot. `MatchLifecycle` loads templates lazily; missing
   templates fall back to color-pixel sampling but template matching is
   substantially more reliable.

When no templates are present, lifecycle detection runs entirely on the
pixel-color samples defined at the top of `matchLifecycle.py`. Those
sample coordinates and reference colors will likely need tuning for
your BlueStacks resolution and theme.
