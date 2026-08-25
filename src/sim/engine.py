"""The simulation core: entities, targeting, combat and match flow.

Everything here is deterministic given a seed. There is no wall clock - time
advances only when :meth:`Simulation.tick` is called - which is what makes the
sim runnable faster than real time and reproducible across runs.

Design notes worth knowing before changing anything:

- Units, buildings and towers are ALL ``Entity``. Towers are just immobile
  entities with a ``tower_key``. Targeting, damage and death then need one
  code path instead of three, and "can this thing shoot that thing" is asked
  the same way everywhere.
- The tick is finer than the observation step (``TICK_DT`` vs 0.25s). Combat
  and movement resolved at 4 Hz would let a fast unit teleport most of a tile
  between frames and skip past its attack range entirely.
- Reinforcement learning will exploit any bug here as though it were a game
  mechanic, so behaviour that is *approximate* is marked as such rather than
  left to look intentional.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from ..game.board import ARENA_COLS, ARENA_ROWS
from . import arena
from .arena import TOWERS, TowerSpec
from .units import (
    STANDARD_LEVEL, Target, UnitStats, stats_at_level, tower_combat,
    unit_elixir_value,
)

# Physics/combat step. 20 Hz: five sub-steps per 0.25s observation cycle.
TICK_DT = 0.05

# Match structure, mirroring GameState so the two agree on what "overtime"
# means. Duplicated as plain numbers rather than imported because GameState
# reads a wall clock and the sim must not.
REGULAR_TIME_END = 180.0
MATCH_MAX_DURATION = 300.0
DOUBLE_ELIXIR_START = 120.0
TRIPLE_ELIXIR_START = 240.0

NORMAL_ELIXIR_RATE = 2.8   # seconds per elixir
DOUBLE_ELIXIR_RATE = 1.4
TRIPLE_ELIXIR_RATE = 0.93

STARTING_ELIXIR = 5.0
MAX_ELIXIR = 10.0

KING_ACTIVATION_HP_FRAC = 0.98

_uid_counter = itertools.count()


@dataclass
class Entity:
    uid: int
    name: str
    friendly: bool
    x: float
    y: float
    hp: float
    max_hp: float
    stats: UnitStats
    deploy_remaining: float = 0.0
    attack_cooldown: float = 0.0
    lifetime_remaining: float = 0.0
    spawn_cooldown: float = 0.0
    target_uid: int | None = None
    tower_key: str | None = None
    active: bool = True          # kings start inactive
    radius: float = 0.4
    # Set once this entity actually starts hitting a building or tower. A
    # troop that has LOCKED ON to a tower keeps hitting it and ignores troops
    # walking up beside it - which is why forcing a retarget in the real game
    # needs a stun or a displacement card, not just a distraction unit.
    locked_on_structure: bool = False
    # Distance walked while pursuing the current target, for charge units.
    charge_distance: float = 0.0
    # Seconds spent firing continuously at the SAME target, for damage ramps.
    fire_time: float = 0.0

    @property
    def charging(self) -> bool:
        return (
            self.stats.charge_range > 0.0
            and self.charge_distance >= self.stats.charge_range
        )

    @property
    def is_tower(self) -> bool:
        return self.tower_key is not None

    @property
    def is_building(self) -> bool:
        return self.stats.is_building or self.is_tower

    @property
    def alive(self) -> bool:
        return self.hp > 0.0

    @property
    def deployed(self) -> bool:
        return self.deploy_remaining <= 0.0

    @property
    def hp_frac(self) -> float:
        return max(0.0, self.hp / self.max_hp) if self.max_hp else 0.0


def _tower_entity(spec: TowerSpec, level: int) -> Entity:
    c = tower_combat(spec.kind, level)
    stats = UnitStats(
        name="kingtower" if spec.is_king else "princesstower",
        hp=c["hp"], damage=c["damage"], hit_speed=c["hit_speed"],
        attack_range=c["attack_range"], speed=0.0, targets=Target.BOTH,
        is_building=True, deploy_time=0.0, aggro_range=c["attack_range"],
        collision_radius=c["collision_radius"],
    )
    return Entity(
        uid=next(_uid_counter), name=stats.name, friendly=spec.friendly,
        x=spec.x, y=spec.y, hp=c["hp"], max_hp=c["hp"], stats=stats,
        tower_key=spec.key, radius=c["collision_radius"],
        # Kings do not fire until a princess falls or they are hit.
        active=not spec.is_king,
    )


def current_damage(e: Entity) -> float:
    """Damage for this swing, accounting for an Inferno-style ramp.

    Stages are cumulative durations of CONTINUOUS fire at one target. Without
    a ramp this is just the flat damage.
    """
    if not e.stats.ramp:
        return e.stats.damage
    elapsed = 0.0
    for duration, damage in e.stats.ramp:
        elapsed += duration
        if e.fire_time < elapsed:
            return damage
    return e.stats.ramp[-1][1]


def can_attack(attacker: Entity, defender: Entity) -> bool:
    """Whether ``attacker`` is allowed to target ``defender`` at all."""
    if attacker.stats.targets is Target.BUILDINGS:
        return defender.is_building
    if defender.stats.flying:
        return attacker.stats.targets in (Target.AIR, Target.BOTH)
    return attacker.stats.targets in (Target.GROUND, Target.BOTH)


@dataclass
class Simulation:
    """One match. Advance with :meth:`tick`; read state off the attributes."""

    seed: int = 0
    # Per-side card and tower LEVELS. Ladder play does not match players
    # exactly, so both sides get their own, and towers can differ from troops
    # (your king tower level and your card levels move independently).
    # The detector cannot read a level off the screen, so this is hidden
    # state the policy has to be robust to rather than condition on.
    friendly_level: int = STANDARD_LEVEL
    enemy_level: int = STANDARD_LEVEL
    friendly_tower_level: int = STANDARD_LEVEL
    enemy_tower_level: int = STANDARD_LEVEL
    # Optional pre-perturbed stat tables (see units.randomize). None means
    # "load the table for this side's level".
    friendly_stats: dict[str, UnitStats] | None = None
    enemy_stats: dict[str, UnitStats] | None = None

    time: float = 0.0
    entities: dict[int, Entity] = field(default_factory=dict)
    elixir: dict[bool, float] = field(default_factory=dict)
    crowns: dict[bool, int] = field(default_factory=dict)
    destroyed_towers: set[str] = field(default_factory=set)
    # Elixir value of THIS side's units that have died. A running total the
    # env diffs per step; it is what makes winning a defensive exchange pay.
    losses: dict[bool, float] = field(default_factory=dict)
    finished: bool = False
    result: str | None = None  # "win" / "loss" / "draw", from friendly's view
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.elixir = {True: STARTING_ELIXIR, False: STARTING_ELIXIR}
        self.crowns = {True: 0, False: 0}
        self.losses = {True: 0.0, False: 0.0}

        f_units, f_spells = stats_at_level(self.friendly_level)
        e_units, e_spells = stats_at_level(self.enemy_level)
        self._stats = {
            True: self.friendly_stats if self.friendly_stats is not None else f_units,
            False: self.enemy_stats if self.enemy_stats is not None else e_units,
        }
        self._spells = {True: f_spells, False: e_spells}
        self._tower_level = {
            True: self.friendly_tower_level,
            False: self.enemy_tower_level,
        }

        for spec in TOWERS:
            e = _tower_entity(spec, self._tower_level[spec.friendly])
            self.entities[e.uid] = e

    @property
    def unit_stats(self) -> dict[str, UnitStats]:
        """The friendly side's table. Convenience for debug and tests; the
        engine itself always goes through ``self._stats[side]``."""
        return self._stats[True]

    # ----- lookups -----

    def towers(self, friendly: bool) -> list[Entity]:
        return [
            e for e in self.entities.values()
            if e.is_tower and e.friendly == friendly and e.alive
        ]

    def tower_hp_fractions(self) -> dict[str, float]:
        """Bar fill per tower key - exactly what the vision layer reports."""
        out = {spec.key: 0.0 for spec in TOWERS}
        for e in self.entities.values():
            if e.is_tower:
                out[e.tower_key] = e.hp_frac
        return out

    def units(self, friendly: bool | None = None) -> list[Entity]:
        return [
            e for e in self.entities.values()
            if not e.is_tower and e.alive and (friendly is None or e.friendly == friendly)
        ]

    # ----- elixir -----

    def elixir_rate(self) -> float:
        if self.time < DOUBLE_ELIXIR_START:
            return NORMAL_ELIXIR_RATE
        if self.time < TRIPLE_ELIXIR_START:
            return DOUBLE_ELIXIR_RATE
        return TRIPLE_ELIXIR_RATE

    def phase(self) -> str:
        if self.time < DOUBLE_ELIXIR_START:
            return "normal"
        if self.time < REGULAR_TIME_END:
            return "double"
        if self.time < TRIPLE_ELIXIR_START:
            return "overtime_double"
        return "overtime_triple"

    # ----- deployment -----

    def can_deploy(self, friendly: bool, name: str, tile_x: int, tile_y: int) -> bool:
        from ..game.cards import get_card_cost

        if self.finished:
            return False
        if self.elixir[friendly] + 1e-6 < get_card_cost(name):
            return False
        return self.is_placeable(friendly, tile_x, tile_y, name=name)

    def is_placeable(
        self, friendly: bool, tile_x: int, tile_y: int, name: str | None = None
    ) -> bool:
        """Whether ``friendly`` may deploy at this ABSOLUTE tile.

        Delegates to ``board.placement_allowed`` - the one implementation of
        the rule. This used to be a second copy, and the two drifted: the
        board unlocked the enemy half on king activation and this did not,
        while this allowed the riverbank row and the board did not. See
        ``placement_allowed`` for what that cost.

        The enemy's tiles are MIRRORED into the placer's frame, so there is
        one coordinate convention and left/right lanes swap with it.

        SPELLS are exempt - they may be cast anywhere. Pass ``name`` so this
        can tell which rule applies; without it the troop rule is assumed,
        which is the safe default for a mask.
        """
        from ..game.board import placement_allowed
        from ..game.cards import is_spell

        spell = name is not None and is_spell(name)
        if friendly:
            tx, ty = tile_x, tile_y
            left_open = "enemy_left" in self.destroyed_towers
            right_open = "enemy_right" in self.destroyed_towers
        else:
            # Mirroring flips x, so the lane a destroyed FRIENDLY left tower
            # opens is on the enemy's right.
            tx = ARENA_COLS - 1 - tile_x
            ty = ARENA_ROWS - 1 - tile_y
            left_open = "friendly_right" in self.destroyed_towers
            right_open = "friendly_left" in self.destroyed_towers
        return placement_allowed(
            tx, ty, left_lane_open=left_open, right_lane_open=right_open,
            spell=spell,
        )

    def deploy(self, friendly: bool, name: str, tile_x: int, tile_y: int) -> bool:
        """Place a card. Returns False if it was not affordable or legal."""
        from ..game.cards import get_card_cost

        if not self.can_deploy(friendly, name, tile_x, tile_y):
            return False
        self.elixir[friendly] -= get_card_cost(name)
        px, py = arena.deploy_position(tile_x, tile_y)

        if name in self._spells[friendly]:
            self._cast_spell(friendly, name, px, py)
            return True

        stats = self._stats[friendly][name]
        for i in range(stats.count):
            # Squads land spread around the point rather than stacked, so
            # splash damage can actually catch more than one of them.
            ox, oy = self._squad_offset(i, stats.count)
            self._spawn(name, friendly, px + ox, py + oy)
        return True

    def _squad_offset(self, i: int, count: int) -> tuple[float, float]:
        if count == 1:
            return 0.0, 0.0
        import math
        angle = 2.0 * math.pi * i / count
        return 0.35 * math.cos(angle), 0.35 * math.sin(angle)

    def _spawn(self, name: str, friendly: bool, x: float, y: float) -> Entity:
        stats = self._stats[friendly][name]
        x, y = arena.clamp_to_arena(x, y)
        e = Entity(
            uid=next(_uid_counter), name=name, friendly=friendly, x=x, y=y,
            hp=stats.hp, max_hp=stats.hp, stats=stats,
            radius=stats.collision_radius,
            deploy_remaining=stats.deploy_time,
            lifetime_remaining=stats.lifetime,
            spawn_cooldown=stats.spawn_period,
        )
        self.entities[e.uid] = e
        return e

    def _cast_spell(self, friendly: bool, name: str, x: float, y: float) -> None:
        damage, radius, building_mult = self._spells[friendly][name]
        for e in list(self.entities.values()):
            if e.friendly == friendly or not e.alive:
                continue
            if arena.distance(x, y, e.x, e.y) > radius:
                continue
            self._damage(e, damage * (building_mult if e.is_building else 1.0))

    # ----- combat -----

    def _damage(self, target: Entity, amount: float) -> None:
        if not target.alive:
            return
        target.hp -= amount
        if target.is_tower and target.stats.name == "kingtower":
            # A king that takes any damage wakes up, which is why chip damage
            # on the king is a real cost and not free.
            if target.hp_frac < KING_ACTIVATION_HP_FRAC:
                target.active = True
        if target.hp <= 0.0:
            self._on_death(target)

    def _on_death(self, e: Entity, expired: bool = False) -> None:
        e.hp = 0.0
        if e.is_tower:
            self._on_tower_destroyed(e)
            return
        if not expired:
            # Expiry is not a kill. A building reaching the end of its
            # lifetime would otherwise hand the other side free reward for
            # doing nothing.
            self.losses[e.friendly] += unit_elixir_value(e.name)
        if e.stats.death_damage > 0.0:
            # Bomb Tower drops a bomb rather than leaving a unit behind.
            for other in list(self.entities.values()):
                if not other.alive or other.friendly == e.friendly:
                    continue
                if arena.distance(e.x, e.y, other.x, other.y) <= e.stats.death_damage_radius:
                    self._damage(other, e.stats.death_damage)
        if e.stats.spawn_on_death:
            for i in range(e.stats.spawn_on_death):
                ox, oy = self._squad_offset(i, max(2, e.stats.spawn_on_death))
                self._spawn(e.stats.spawns, e.friendly, e.x + ox, e.y + oy)

    def _on_tower_destroyed(self, tower: Entity) -> None:
        self.destroyed_towers.add(tower.tower_key)
        scorer = not tower.friendly
        if tower.stats.name == "kingtower":
            self.crowns[scorer] = 3
            self._finish()
            return
        self.crowns[scorer] += 1
        # Losing a princess activates your king.
        for other in self.towers(tower.friendly):
            if other.stats.name == "kingtower":
                other.active = True
        if self.time >= REGULAR_TIME_END:
            # Overtime is sudden death: the first tower to fall ends it.
            self._finish()

    def _finish(self) -> None:
        self.finished = True
        mine, theirs = self.crowns[True], self.crowns[False]
        if mine > theirs:
            self.result = "win"
        elif theirs > mine:
            self.result = "loss"
        else:
            self.result = self._tiebreak()

    def _tiebreak(self) -> str:
        """Equal crowns at time: lowest remaining tower HP loses."""
        mine = min((t.hp_frac for t in self.towers(True)), default=0.0)
        theirs = min((t.hp_frac for t in self.towers(False)), default=0.0)
        if abs(mine - theirs) < 1e-6:
            return "draw"
        return "win" if mine > theirs else "loss"

    # ----- targeting -----

    # Hysteresis on the leash: a troop target is dropped only past this
    # multiple of sight range, so a unit at the boundary does not flip
    # between chasing and walking away every tick.
    LEASH_FACTOR = 1.25

    def _enemies_of(self, e: Entity) -> list[Entity]:
        return [
            o for o in self.entities.values()
            if o.alive and o.friendly != e.friendly and can_attack(e, o)
        ]

    def _nearest_troop_in_sight(self, e: Entity) -> Entity | None:
        """Nearest attackable NON-tower enemy inside sight range.

        Buildings other than towers count: a ground troop can hit a Goblin
        Hut, so one standing in its path is a legitimate distraction.
        Building-only attackers have ``aggro_range`` zeroed and so never
        divert, which is what makes a Giant walk past a Musketeer.
        """
        if e.stats.aggro_range <= 0.0:
            return None
        candidates = [
            o for o in self._enemies_of(e)
            if not o.is_tower
            and arena.distance(e.x, e.y, o.x, o.y) <= e.stats.aggro_range
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda o: arena.distance(e.x, e.y, o.x, o.y))

    def _default_goal(self, e: Entity) -> Entity | None:
        """Where this entity heads with nothing else to fight.

        Normal troops advance on the nearest enemy CROWN TOWER. Building-only
        attackers instead take the nearest building of any kind, which is how
        a hut placed in front of the towers pulls a Giant off them.
        """
        enemies = self._enemies_of(e)
        if not enemies:
            return None
        if e.stats.targets is Target.BUILDINGS:
            pool = [o for o in enemies if o.is_building]
        else:
            pool = [o for o in enemies if o.is_tower]
        if not pool:
            pool = enemies
        return min(pool, key=lambda o: arena.distance(e.x, e.y, o.x, o.y))

    def _acquire_target(self, e: Entity) -> Entity | None:
        """First target: nearest enemy in sight, else the default goal."""
        return self._nearest_troop_in_sight(e) or self._default_goal(e)

    def _retarget(self, e: Entity) -> Entity | None:
        """Per-tick target maintenance.

        The rule this encodes, from how the real game behaves:

        - A troop target is HELD until it dies or runs beyond the leash. Units
          do not shop around for a better target mid-fight.
        - A structure target is held only while the unit is still WALKING to
          it. A troop entering sight range diverts the unit - that is ordinary
          distraction, and it works.
        - Once the unit has actually started hitting a structure
          (``locked_on_structure``), it is committed and no longer diverts.
          This is why a Royal Giant on your tower keeps hitting the tower and
          has to be pushed or stunned off it rather than merely distracted.
        """
        target = self.entities.get(e.target_uid) if e.target_uid else None

        if target is not None and (not target.alive or not can_attack(e, target)):
            target = None
            e.locked_on_structure = False

        if target is not None and not target.is_building:
            leash = e.stats.aggro_range * self.LEASH_FACTOR
            if leash > 0 and arena.distance(e.x, e.y, target.x, target.y) > leash:
                target = None

        if target is None:
            e.locked_on_structure = False
            target = self._acquire_target(e)
        elif target.is_building and not e.locked_on_structure:
            target = self._nearest_troop_in_sight(e) or target

        if target is not None and target.uid != e.target_uid:
            # A ramp is earned against ONE target and resets on a switch -
            # that is the whole counterplay to Inferno Tower. Charge resets
            # too: a unit that turns has to build its run up again.
            e.fire_time = 0.0
            e.charge_distance = 0.0
        e.target_uid = target.uid if target else None
        return target

    # ----- the tick -----

    def tick(self, dt: float = TICK_DT) -> None:
        if self.finished:
            return

        self.time += dt
        rate = self.elixir_rate()
        for side in (True, False):
            self.elixir[side] = min(MAX_ELIXIR, self.elixir[side] + dt / rate)

        for e in list(self.entities.values()):
            if not e.alive:
                continue
            if e.deploy_remaining > 0.0:
                e.deploy_remaining -= dt
                continue
            if e.attack_cooldown > 0.0:
                e.attack_cooldown -= dt
            self._tick_building(e, dt)
            if not e.alive:
                continue
            self._tick_combat(e, dt)

        self.entities = {
            uid: e for uid, e in self.entities.items() if e.alive or e.is_tower
        }
        self._check_time()

    def _tick_building(self, e: Entity, dt: float) -> None:
        if not e.stats.is_building or e.is_tower:
            return
        if e.stats.lifetime:
            e.lifetime_remaining -= dt
            if e.lifetime_remaining <= 0.0:
                self._on_death(e, expired=True)
                return
        if e.stats.spawn_period and e.stats.spawns:
            e.spawn_cooldown -= dt
            if e.spawn_cooldown <= 0.0:
                e.spawn_cooldown += e.stats.spawn_period
                for i in range(e.stats.spawn_count):
                    ox, oy = self._squad_offset(i, max(2, e.stats.spawn_count))
                    self._spawn(e.stats.spawns, e.friendly, e.x + ox, e.y + oy)

    def _tick_combat(self, e: Entity, dt: float) -> None:
        if e.is_tower and not e.active:
            return

        target = self._retarget(e)
        if target is None:
            return

        # Range is measured between HITBOX EDGES, not centres, so both radii
        # come off the gap. Omitting the attacker's radius let melee units
        # stand inside each other.
        reach = e.stats.attack_range + target.radius + e.radius
        dist = arena.distance(e.x, e.y, target.x, target.y)

        if dist <= reach:
            e.fire_time += dt
            if e.attack_cooldown <= 0.0 and e.stats.damage > 0.0:
                self._attack(e, target)
            return

        # Out of reach: not firing, so a ramp decays back to its first stage.
        e.fire_time = 0.0

        if e.stats.speed > 0.0:
            self._move_toward(e, target, dt)
        else:
            # Immobile and out of reach: drop the target so it re-acquires
            # something it can actually hit next tick.
            e.target_uid = None

    def _attack(self, e: Entity, target: Entity) -> None:
        e.attack_cooldown = e.stats.hit_speed
        if target.is_building:
            # Committed now: see _retarget.
            e.locked_on_structure = True

        damage = current_damage(e)
        self._damage(target, damage)
        if e.stats.splash_radius > 0.0:
            for other in list(self.entities.values()):
                if other.uid == target.uid or not other.alive:
                    continue
                if other.friendly == e.friendly or not can_attack(e, other):
                    continue
                if arena.distance(target.x, target.y, other.x, other.y) <= e.stats.splash_radius:
                    self._damage(other, damage)

        # A charge is spent on impact; the unit has to build another run up.
        e.charge_distance = 0.0

        if e.stats.kamikaze:
            # Spirits and the Battle Ram land one hit and are gone. Routed
            # through _on_death so their death spawn still fires - that is
            # how a Battle Ram becomes two Barbarians.
            self._on_death(e)

    def _move_toward(self, e: Entity, target: Entity, dt: float) -> None:
        if e.stats.flying:
            gx, gy = target.x, target.y
        else:
            gx, gy = arena.ground_waypoint(e.x, e.y, e.friendly, target.x, target.y)

        dx, dy = gx - e.x, gy - e.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-6:
            return
        step = e.stats.speed * dt
        if e.stats.charge_range > 0.0:
            if e.charging:
                step *= e.stats.charge_speed_mult or 1.0
            e.charge_distance += step
        nx = e.x + dx / dist * step
        ny = e.y + dy / dist * step

        if not e.stats.flying and arena.blocks_ground(nx, ny):
            # Walked into water: slide along the bank toward the bridge at
            # full speed instead of stopping dead. Previously this moved a
            # FRACTION of the remaining distance per tick, which eased to a
            # crawl near the bridge and looked broken.
            bridge_x = arena.nearest_bridge_x(e.x)
            direction = 1.0 if bridge_x > e.x else -1.0
            nx = e.x + direction * min(step, abs(bridge_x - e.x))
            ny = e.y
        e.x, e.y = arena.clamp_to_arena(nx, ny)

    def _check_time(self) -> None:
        if self.finished:
            return
        if self.time >= REGULAR_TIME_END and self.crowns[True] != self.crowns[False]:
            self._finish()
            return
        if self.time >= MATCH_MAX_DURATION:
            self._finish()
