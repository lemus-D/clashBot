"""Detector class manifest: which units and cards the observation encodes.

This module owns the NAMING CONTRACT. Everything that needs to know "what
troops exist" imports it from here, so there is exactly one list and it
cannot drift from the detector.

Why a generated manifest instead of a hand-written tuple
--------------------------------------------------------
A hand-written list has silently broken this project twice. Plural entries
("minions" vs the model's "minion") meant four of the most common units
never entered the arena tensor at all, and later eight arena-2 names were
added that ``troop-counter/8`` cannot emit, making 2304 observation values
permanently zero. Both were invisible: the arena just looked empty.

So the list is DERIVED from ``get_model(...).class_names`` and written to
``class_manifest.json`` by ``python -m src.main --derive-classes``. Runtime
reads the manifest - deriving at import would put a network call and an API
key in the path of every test, every offline tool and the simulator.
``verify_against_model`` closes the loop: ``ClashEnv`` calls it once the real
model is loaded and raises if the manifest has gone stale.

Two index spaces, not one
-------------------------
Arena classes and card classes are OVERLAPPING SETS where neither contains
the other, so they get separate index spaces:

- Goblin Brawler is an arena unit with no card - it spawns from Goblin Cage.
  The model has ``blue goblin brawler`` but no ``card goblin brawler``.
- The same is true of every spawn-only unit (Lava Pups, Golemites, ...), so
  this is a category, not a special case.

Collapsing them into one list gave the hand one-hot a channel for a card
that can never be in hand. Keeping them separate means a future spawn-only
unit needs no special-casing at all: it lands in ARENA_CLASSES and not in
CARD_CLASSES, automatically.

Ordering is APPEND-ONLY
-----------------------
Class order is channel order, and channel order is baked into every
recording and every trained checkpoint. ``merge_preserving_order`` keeps
existing names at their existing index and appends new ones at the end, so
old index i is always new index i. That is what makes it possible to grow a
trained network's input layer by scattering the old weight columns into
their new positions instead of retraining from scratch. Sorting the list -
or taking the model's own order, which is not guaranteed stable across
versions - would reshuffle every channel and make that impossible.
"""

from __future__ import annotations

import json
import os

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "class_manifest.json")

# Arena classes that are real but deliberately not observation channels.
# TOWER SKINS change how towers look, so these two detect unreliably. That
# also rules out reading a princess tower vanishing from detections as a
# destruction signal - it inherits the same unreliability. Tower state comes
# from the HP bar (``src/vision/towers.py``) instead, and towers are static,
# so channels for them would carry no information anyway.
IGNORED_ARENA_CLASSES: frozenset[str] = frozenset({"kingtower", "princesstower"})

_BLUE_PREFIX = "blue "
_RED_PREFIX = "red "
_CARD_PREFIX = "card "


def normalize_name(name: str) -> str:
    """Canonical card/troop key: lowercase, no spaces/underscores/hyphens."""
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def derive_from_class_names(class_names) -> tuple[list[str], list[str]]:
    """Split raw detector class names into (arena_classes, card_classes).

    Names are ``blue <unit>`` / ``red <unit>`` / ``card <name>``. Anything
    else is a contract change and raises rather than being skipped: an
    unrecognised prefix means this parser no longer understands the model.

    Returned lists are sorted, which is fine here - callers put them through
    ``merge_preserving_order`` against the existing manifest to get the
    stable append-only order.
    """
    blue: set[str] = set()
    red: set[str] = set()
    cards: set[str] = set()
    unparsed: list[str] = []

    for raw in class_names:
        lowered = raw.lower()
        if lowered.startswith(_BLUE_PREFIX):
            blue.add(normalize_name(lowered[len(_BLUE_PREFIX):]))
        elif lowered.startswith(_RED_PREFIX):
            red.add(normalize_name(lowered[len(_RED_PREFIX):]))
        elif lowered.startswith(_CARD_PREFIX):
            cards.add(normalize_name(lowered[len(_CARD_PREFIX):]))
        else:
            unparsed.append(raw)

    if unparsed:
        raise ValueError(
            f"Detector class names {unparsed!r} have none of the expected "
            f"prefixes ('blue ', 'red ', 'card '). The model naming "
            f"convention changed; update derive_from_class_names rather "
            f"than dropping these, or they become silent holes in the "
            f"observation."
        )

    if blue != red:
        raise ValueError(
            f"Detector emits different units per side, which breaks the "
            f"friendly/enemy channel split: blue-only={sorted(blue - red)!r} "
            f"red-only={sorted(red - blue)!r}. The arena tensor packs both "
            f"sides over one class list and assumes they match."
        )

    arena = sorted(blue - IGNORED_ARENA_CLASSES)
    return arena, sorted(cards)


def merge_preserving_order(existing: list[str], derived: list[str]) -> list[str]:
    """Append-only merge: existing names keep their index, new ones go last.

    A name that has DISAPPEARED from the model is kept, not dropped.
    Dropping it would shift every later index and silently invalidate the
    channel meanings of existing recordings and checkpoints; keeping it
    costs one dead channel. Callers should surface these - see the
    ``--derive-classes`` output.
    """
    out = list(existing)
    seen = set(out)
    for name in derived:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Class manifest {path} is missing. It is generated from the "
            f"detector, not written by hand: run "
            f"`python -m src.main --derive-classes`."
        )
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ("model_id", "arena_classes", "card_classes"):
        if key not in manifest:
            raise ValueError(f"Class manifest {path} has no {key!r} key.")
    return manifest


def write_manifest(
    model_id: str,
    arena_classes: list[str],
    card_classes: list[str],
    path: str = MANIFEST_PATH,
) -> None:
    blob = {
        "_comment": (
            "GENERATED by `python -m src.main --derive-classes`. Do not edit "
            "by hand. Order is append-only and is the observation channel "
            "order - reordering invalidates every recording and checkpoint."
        ),
        "model_id": model_id,
        "arena_classes": arena_classes,
        "card_classes": card_classes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
        f.write("\n")


_manifest = load_manifest()

MODEL_ID: str = _manifest["model_id"]
ARENA_CLASSES: tuple[str, ...] = tuple(_manifest["arena_classes"])
CARD_CLASSES: tuple[str, ...] = tuple(_manifest["card_classes"])

ARENA_INDEX: dict[str, int] = {n: i for i, n in enumerate(ARENA_CLASSES)}
CARD_INDEX: dict[str, int] = {n: i for i, n in enumerate(CARD_CLASSES)}


def verify_against_model(class_names, model_id: str) -> None:
    """Raise if the live model disagrees with the manifest.

    Called once when the real detector loads. Without this the manifest is
    just another hand-maintained list that happens to have been generated
    once - the whole point is that drift is impossible, and drift is only
    impossible if something actually checks.
    """
    arena, cards = derive_from_class_names(class_names)
    missing_arena = [n for n in arena if n not in ARENA_INDEX]
    missing_cards = [n for n in cards if n not in CARD_INDEX]
    if missing_arena or missing_cards:
        raise ValueError(
            f"Model {model_id!r} emits classes the manifest does not know: "
            f"arena={missing_arena!r} cards={missing_cards!r}. These would be "
            f"dropped from every observation. Regenerate with "
            f"`python -m src.main --derive-classes` (this changes the "
            f"observation schema hash and invalidates existing recordings "
            f"and checkpoints)."
        )
