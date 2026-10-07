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

# Per-level arrays begin at the card's OWN level 1, and which DISPLAYED level
# that is depends on rarity: a Rare's level 1 is displayed level 3, an Epic's
# is 6, a Legendary's is 9. index = displayed_level - FIRST_LEVEL[rarity].
# Mixing this up silently blends power levels across cards, which is the exact
# bug this script was written to fix.
FIRST_LEVEL = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9, "Champion": 11}

# Displayed levels emitted into the manifest. Ladder play around tournament
# standard (11) sees roughly this band, and the simulator samples within it so
# a policy cannot assume a fixed power level - the detector cannot read levels
# off the screen, so the policy must be robust to not knowing.
LEVELS = range(9, 15)
STANDARD_LEVEL = 11

# 1 CR tile = 1000 range units; 1 grid tile = 2 CR tiles.
RANGE_TO_GRID = 2000.0
# Speed is in CR tiles/minute (45 slow, 60 medium, 90 fast, 120 very fast).
SPEED_TO_GRID = 120.0
MS = 1000.0

# our key -> game-data character name.
#
# Cards beyond arena 1 are STAGED: modelled here so the simulator is ready,
# but NOT added to the detector's class manifest, because troop-counter/8
# cannot emit them. Putting them in the manifest early is precisely the bug
# that was reverted in 309117a - every one becomes a permanently-zero
# observation channel. When the model gains them, `--derive-classes` picks
# them up and these stats are already waiting.
UNITS = {
    # TrainingCamp + Arena 1: what the detector can see today.
    "knight": "Knight",
    "archer": "Archer",
    "minion": "Minion",
    "goblin": "Goblin",
    "speargoblin": "SpearGoblin",
    "musketeer": "Musketeer",
    "minipekka": "MiniPekka",
    "giant": "Giant",
    "goblinbrawler": "GoblinBrawler",
    # Arena 2
    "skeleton": "Skeleton",
    "valkyrie": "Valkyrie",
    "bomber": "Bomber",
    # Arena 3
    "barbarian": "Barbarian",
    "battleram": "BattleRam",
    "megaminion": "MegaMinion",
    # Arena 4
    "wizard": "Wizard",
    "firespirit": "FireSpirits",
    "electrospirit": "ElectroSpirit",
    "skeletondragon": "SkeletonDragon",
}
BUILDINGS = {
    "goblincage": "GoblinCage",
    "goblinhut": "GoblinHut",
    # Arena 2-4
    "tombstone": "Tombstone",
    "cannon": "Cannon",
    "infernotower": "InfernoTower",
    "bombtower": "BombTower",
}
SPELLS = {"arrows": "ArrowsSpell", "fireball": "FireballSpell"}

# Everything the current detector CANNOT emit. Kept out of the manifest.
STAGED = {
    "skeleton", "valkyrie", "bomber", "tombstone",
    "barbarian", "battleram", "megaminion", "cannon",
    "wizard", "firespirit", "electrospirit", "skeletondragon",
    "infernotower", "bombtower",
}

# Squad sizes, from the card entries' summon_number (0 means 1).
COUNTS = {
    "archer": 2, "minion": 3, "goblin": 3, "speargoblin": 3,
    "skeleton": 3, "barbarian": 5, "skeletondragon": 2,
}

# Units whose death releases something, where the spawned character is not
# one of ours under the same name.
DEATH_SPAWN_MAP = {
    "SpearGoblin": "speargoblin",
    "GoblinBrawler": "goblinbrawler",
    "Barbarian": "barbarian",
    "Skeleton": "skeleton",
}


def fetch(name: str) -> list[dict]:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.json")
    if not os.path.exists(path):
        url = f"{BASE}/{FILES[name]}.json"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def at_level(entry: dict, field: str, rarity: str, level: int = STANDARD_LEVEL):
    """Value of ``field`` at a DISPLAYED card level."""
    arr = entry.get(f"{field}_per_level")
    if not arr:
        return entry.get(field)
    idx = level - FIRST_LEVEL.get(rarity, 1)
    return arr[max(0, min(idx, len(arr) - 1))]


def by_level(entry: dict, field: str, rarity: str) -> dict[str, int]:
    """``{displayed_level: value}`` across the emitted band.

    Only HP and damage scale with level in Clash Royale - speed, range and
    hit speed do not - so those are the only fields that get a table.
    """
    # 0 rather than None for things that simply do not have the stat - a
    # spawner has no damage, and a null would have to be special-cased by
    # every reader.
    return {str(L): (at_level(entry, field, rarity, L) or 0) for L in LEVELS}


def splash_of(entry: dict, proj: dict) -> float:
    """Area-damage radius in grid tiles, 0 for single-target.

    Melee splashers (Valkyrie) carry it on the character as
    ``area_damage_radius``; ranged ones (Bomber, Wizard) carry it on their
    projectile's ``radius``. A projectile with no radius is a single-target
    shot, not an area one - Mega Minion and Cannon land here.
    """
    r = entry.get("area_damage_radius")
    if not r and proj:
        r = proj.get("radius")
    return round((r or 0) / RANGE_TO_GRID, 4)


def ramp_of(entry: dict, rarity: str) -> list:
    """Inferno-style damage ramp as ``[[seconds, damage], ...]``.

    The game data gives the later stages only at level 1
    (``variable_damage2``/``3``), so they are scaled by the same factor the
    base damage moves by. Derived, not measured - flagged in the output.
    """
    t1 = entry.get("variable_damage_time1")
    if not t1:
        return []
    base1 = entry.get("damage") or 0
    scaled = at_level(entry, "damage", rarity) or 0
    factor = (scaled / base1) if base1 else 1.0
    stages = [[t1 / MS, scaled]]
    for n in (2, 3):
        d = entry.get(f"variable_damage{n}")
        if not d:
            break
        dur = entry.get(f"variable_damage_time{n}", 0) / MS
        stages.append([dur, round(d * factor, 1)])
    return stages


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

        dmg_src = c if at_level(c, "damage", rarity) else pr.get(c.get("projectile"), {})
        units[key] = {
            "hp": at_level(c, "hitpoints", rarity),
            "damage": damage or 0,
            "hp_by_level": by_level(c, "hitpoints", rarity),
            "damage_by_level": by_level(dmg_src, "damage", rarity),
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
            "splash_radius": splash_of(c, pr.get(c.get("projectile")) or {}),
            "kamikaze": bool(c.get("kamikaze")),
            "charge_range": round((c.get("charge_range") or 0) / RANGE_TO_GRID, 4),
            "charge_speed_mult": (c.get("charge_speed_multiplier") or 0) / 100.0,
            "spawn_on_death": c.get("death_spawn_count") or 0,
            "spawns": DEATH_SPAWN_MAP.get(c.get("death_spawn_character")),
            "staged": key in STAGED,
            "source_name": gname,
            "source_rarity": rarity,
        }

    for key, gname in BUILDINGS.items():
        b = bd.get(gname)
        if b is None:
            raise SystemExit(f"building {gname!r} not in game data")
        rarity = b.get("rarity") or "Rare"
        spawn = b.get("spawn_character")
        # Defensive buildings carry their damage on a projectile, exactly as
        # ranged troops do; spawners genuinely have none.
        bproj = pr.get(b.get("projectile")) or {}
        dmg_src = b if at_level(b, "damage", rarity) else bproj

        # A death "spawn" that is not a real unit is an explosion. Bomb Tower
        # drops a BombTowerBomb, which is a one-off area hit rather than
        # something that walks around, so it is modelled as death damage.
        death_char = b.get("death_spawn_character")
        bomb = bd.get(death_char) if death_char else None
        death_damage = death_damage_radius = 0.0
        if bomb and bomb.get("death_damage"):
            death_damage = at_level(bomb, "death_damage", rarity) or bomb["death_damage"]
            death_damage_radius = round(
                (bomb.get("death_damage_radius") or 0) / RANGE_TO_GRID, 4
            )
            death_char = None  # consumed as damage, not as a spawn

        units[key] = {
            "hp": at_level(b, "hitpoints", rarity),
            "damage": at_level(dmg_src, "damage", rarity) or 0,
            "hp_by_level": by_level(b, "hitpoints", rarity),
            "damage_by_level": by_level(dmg_src, "damage", rarity),
            "death_damage": death_damage,
            "death_damage_radius": death_damage_radius,
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
            "spawn_on_death": (b.get("death_spawn_count") or 0) if death_char else 0,
            "spawns": DEATH_SPAWN_MAP.get(death_char or spawn),
            "splash_radius": splash_of(b, pr.get(b.get("projectile")) or {}),
            "kamikaze": False,
            "charge_range": 0.0,
            "charge_speed_mult": 0.0,
            "ramp": ramp_of(b, rarity),
            "staged": key in STAGED,
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
            "damage_by_level": by_level(p, "damage", rarity),
            "radius": round((p.get("radius") or 2000) / RANGE_TO_GRID, 4),
            "tower_damage_mult": round(mult, 4),
            "source_name": pname,
            "source_rarity": rarity,
        }

    towers: dict[str, dict] = {}
    for key, gname in (("princess", "PrincessTower"), ("king", "KingTower")):
        b = bd.get(gname)
        if b is None:
            raise SystemExit(f"tower {gname!r} not in game data")
        rarity = b.get("rarity") or "Common"
        proj = pr.get(b.get("projectile")) or {}
        towers[key] = {
            "hp_by_level": by_level(b, "hitpoints", rarity),
            "damage_by_level": by_level(proj, "damage", rarity),
            "hit_speed": (b.get("hit_speed") or 1000) / MS,
            "attack_range": round((b.get("range") or 7000) / RANGE_TO_GRID, 4),
            "collision_radius": round(
                (b.get("collision_radius") or 1000) / RANGE_TO_GRID, 4
            ),
            "source_name": gname,
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
        "levels": list(LEVELS),
        "standard_level": STANDARD_LEVEL,
        "units": units,
        "spells": spells,
        "towers": towers,
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
    for k, t in sorted(towers.items()):
        print(f"{k+' tower':16s} {t['hp_by_level'][str(STANDARD_LEVEL)]:6} hp  "
              f"{t['damage_by_level'][str(STANDARD_LEVEL)]:4} dmg  "
              f"(lvl {min(LEVELS)}-{max(LEVELS)} available)")
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
