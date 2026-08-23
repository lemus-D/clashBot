"""Generate ``src/sim/unit_stats.json`` from real Clash Royale game data.

    python tools/derive_unit_stats.py

The simulator's stat table used to be written from recollection, which was
wrong in a way that mattered: the values turned out to be roughly LEVEL 1
figures, and level 1 is not comparable across rarities, so relative unit
strength - the thing a policy actually learns - was distorted. This script
replaces that with numbers pulled from RoyaleAPI's cr-api-data, which is
extracted from the game client itself.

Source: https://royaleapi.github.io/cr-api-data/ (APK-derived, community
maintained). Downloaded to ``.crdata/`` (gitignored); only the small derived
manifest is checked in.

Everything is normalised to TOURNAMENT STANDARD - displayed level 11. The
per-level arrays start at each card's own level 1, and a card's first level
depends on rarity, so the index differs per rarity. Getting that wrong is the
easiest way to reintroduce exactly the bug this script exists to fix.

Distances are converted from the real game's 18x32 arena to this project's
9x16 grid, which means HALVING them.
"""

from __future__ import annotations

import json
import os
import urllib.request

BASE = "https://royaleapi.github.io/cr-api-data/json"
CACHE = os.path.join(os.path.dirname(__file__), "..", ".crdata")
OUT = os.path.join(
    os.path.dirname(__file__), "..", "src", "sim", "unit_stats.json"
)

FILES = {
    "characters": "cards_stats_characters",
    "building": "cards_stats_building",
    "projectile": "cards_stats_projectile",
    "troop": "cards_stats_troop",
    "cards": "cards",
}

# Displayed level 11 in per-level arrays that begin at the card's OWN level 1.
# A Rare's level 1 is displayed level 3, an Epic's is 6, a Legendary's is 9.
LEVEL_INDEX = {"Common": 10, "Rare": 8, "Epic": 5, "Legendary": 2, "Champion": 2}

# 1 CR tile = 1000 range units; 1 grid tile = 2 CR tiles.
RANGE_TO_GRID = 2000.0
# Speed is in CR tiles/minute (45 slow, 60 medium, 90 fast, 120 very fast).
SPEED_TO_GRID = 120.0
MS = 1000.0

# our ARENA_CLASSES key -> game-data character name
UNITS = {
    "knight": "Knight",
    "archer": "Archer",
    "minion": "Minion",
    "goblin": "Goblin",
    "speargoblin": "SpearGoblin",
    "musketeer": "Musketeer",
    "minipekka": "MiniPekka",
    "giant": "Giant",
    "goblinbrawler": "GoblinBrawler",
}
BUILDINGS = {"goblincage": "GoblinCage", "goblinhut": "GoblinHut"}
SPELLS = {"arrows": "ArrowsSpell", "fireball": "FireballSpell"}

# Squad sizes. summon_number in the card data is unreliable for these, so they
# are stated here and asserted against the deployed-count behaviour in tests.
COUNTS = {"archer": 2, "minion": 3, "goblin": 3, "speargoblin": 3}


def fetch(name: str) -> list[dict]:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.json")
    if not os.path.exists(path):
        url = f"{BASE}/{FILES[name]}.json"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def at_level(entry: dict, field: str, rarity: str):
    """Tournament-standard value from a per-level array."""
    arr = entry.get(f"{field}_per_level")
    if not arr:
        return entry.get(field)
    idx = LEVEL_INDEX.get(rarity, 10)
    return arr[min(idx, len(arr) - 1)]


def targets_of(entry: dict) -> str:
    if entry.get("target_only_buildings"):
        return "buildings"
    air = bool(entry.get("attacks_air"))
    ground = bool(entry.get("attacks_ground"))
    if air and ground:
        return "both"
    return "air" if air else "ground"


def main() -> None:
    print("Loading game data")
    ch = {c.get("name"): c for c in fetch("characters")}
    bd = {b.get("name"): b for b in fetch("building")}
    pr = {p.get("name"): p for p in fetch("projectile")}
    cards = {c.get("key"): c for c in fetch("cards")}

    units: dict[str, dict] = {}

    for key, gname in UNITS.items():
        c = ch.get(gname)
        if c is None:
            raise SystemExit(f"character {gname!r} not in game data")
        rarity = c.get("rarity") or "Common"

        damage = at_level(c, "damage", rarity)
        if not damage and c.get("projectile"):
            # Ranged units carry their damage on the projectile, not the body.
            p = pr.get(c["projectile"])
            if p:
                damage = at_level(p, "damage", rarity)

        units[key] = {
            "hp": at_level(c, "hitpoints", rarity),
            "damage": damage or 0,
            "hit_speed": (c.get("hit_speed") or 1000) / MS,
            "attack_range": round((c.get("range") or 500) / RANGE_TO_GRID, 4),
            "speed": round((c.get("speed") or 60) / SPEED_TO_GRID, 4),
            "targets": targets_of(c),
            "flying": bool(c.get("flying_height")),
            "count": COUNTS.get(key, 1),
            "deploy_time": (c.get("deploy_time") or 1000) / MS,
            "aggro_range": round((c.get("sight_range") or 5500) / RANGE_TO_GRID, 4),
            "collision_radius": round(
                (c.get("collision_radius") or 500) / RANGE_TO_GRID, 4
            ),
            "is_building": False,
            "source_name": gname,
            "source_rarity": rarity,
        }

    for key, gname in BUILDINGS.items():
        b = bd.get(gname)
        if b is None:
            raise SystemExit(f"building {gname!r} not in game data")
        rarity = b.get("rarity") or "Rare"
        spawn = b.get("spawn_character")
        units[key] = {
            "hp": at_level(b, "hitpoints", rarity),
            "damage": at_level(b, "damage", rarity) or 0,
            "hit_speed": (b.get("hit_speed") or 1000) / MS,
            "attack_range": round((b.get("range") or 0) / RANGE_TO_GRID, 4),
            "speed": 0.0,
            "targets": targets_of(b),
            "flying": False,
            "count": 1,
            "deploy_time": (b.get("deploy_time") or 1000) / MS,
            "aggro_range": round((b.get("sight_range") or 5500) / RANGE_TO_GRID, 4),
            "collision_radius": round(
                (b.get("collision_radius") or 1000) / RANGE_TO_GRID, 4
            ),
            "is_building": True,
            "lifetime": (b.get("life_time") or 0) / MS,
            # A hut releases a batch every spawn_pause_time; spawn_interval is
            # the gap WITHIN a batch, which this sim does not model.
            "spawn_period": (b.get("spawn_pause_time") or 0) / MS,
            "spawn_count": b.get("spawn_number") or 0,
            "spawn_on_death": b.get("death_spawn_count") or 0,
            "spawns": {"SpearGoblin": "speargoblin",
                       "GoblinBrawler": "goblinbrawler"}.get(
                           b.get("death_spawn_character") or spawn),
            "source_name": gname,
            "source_rarity": rarity,
        }

    spells: dict[str, dict] = {}
    for key, pname in SPELLS.items():
        p = pr.get(pname)
        if p is None:
            raise SystemExit(f"projectile {pname!r} not in game data")
        rarity = (cards.get(key) or {}).get("rarity") or "Common"
        # crown_tower_damage_percent is a REDUCTION expressed as a negative
        # percentage: -70 means a crown tower takes 30% of the damage.
        crown = p.get("crown_tower_damage_percent")
        mult = 1.0 + (crown / 100.0) if crown is not None else 0.30
        spells[key] = {
            "damage": at_level(p, "damage", rarity),
            "radius": round((p.get("radius") or 2000) / RANGE_TO_GRID, 4),
            "tower_damage_mult": round(mult, 4),
            "source_name": pname,
            "source_rarity": rarity,
        }

    blob = {
        "_comment": (
            "GENERATED by tools/derive_unit_stats.py from RoyaleAPI cr-api-data "
            "(APK-derived). Do not edit by hand. Values are TOURNAMENT STANDARD "
            "(displayed level 11). Distances are GRID tiles: real-game tiles "
            "halved for this project's 9x16 arena. Times are seconds."
        ),
        "source": BASE,
        "level": "tournament standard (displayed 11)",
        "units": units,
        "spells": spells,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nwrote {OUT}")
    print(f"{'unit':16s} {'hp':>6s} {'dmg':>5s} {'hit':>5s} {'range':>6s} "
          f"{'speed':>6s} {'targets':>9s} {'n':>2s}")
    for k, u in sorted(units.items()):
        print(f"{k:16s} {u['hp']:6} {u['damage']:5} {u['hit_speed']:5.2f} "
              f"{u['attack_range']:6.2f} {u['speed']:6.3f} {u['targets']:>9s} "
              f"{u['count']:2}")
    for k, s in sorted(spells.items()):
        print(f"{k:16s} {s['damage']:6} dmg  radius {s['radius']:.2f}  "
              f"tower x{s['tower_damage_mult']:.2f}")
    print(
        "\nNOTE: Arrows radius comes out at 1.4 real-game tiles, where the "
        "commonly cited figure is 4.0. Fireball's 2.5 matches exactly, so the "
        "conversion is right and the Arrows field probably describes a single "
        "arrow rather than the volley spread. Left as the data says, flagged "
        "here rather than silently overridden."
    )


if __name__ == "__main__":
    main()
