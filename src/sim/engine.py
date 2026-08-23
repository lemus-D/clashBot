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
from .units import SPELL_DAMAGE, UNIT_STATS, Target, UnitStats

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


def _tower_entity(spec: TowerSpec) -> Entity:
    stats = UnitStats(
        name="kingtower" if spec.is_king else "princesstower",
        hp=spec.max_hp, damage=spec.damage, hit_speed=spec.hit_speed,
        attack_range=spec.attack_range, speed=0.0, targets=Target.BOTH,
        is_building=True, deploy_time=0.0, aggro_range=spec.attack_range,
    )
    return Entity(
        uid=next(_uid_counter), name=stats.name, friendly=spec.friendly,
        x=spec.x, y=spec.y, hp=spec.max_hp, max_hp=spec.max_hp, stats=stats,
        tower_key=spec.key, radius=spec.radius,
        # Kings do not fire until a princess falls or they are hit.
        active=not spec.is_king,
    )


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
    unit_stats: dict[str, UnitStats] = field(default_factory=lambda: dict(UNIT_STATS))

    time: float = 0.0
    entities: dict[int, Entity] = field(default_factory=dict)
    elixir: dict[bool, float] = field(default_factory=dict)
    crowns: dict[bool, int] = field(default_factory=dict)
    destroyed_towers: set[str] = field(default_factory=set)
    finished: bool = False
    result: str | None = None  # "win" / "loss" / "draw", from friendly's view
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.elixir = {True: STARTING_ELIXIR, False: STARTING_ELIXIR}
        self.crowns = {True: 0, False: 0}
        for spec in TOWERS:
            e = _tower_entity(spec)
            self.entities[e.uid] = e

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
        return self.is_placeable(friendly, tile_x, tile_y)

    def is_placeable(self, friendly: bool, tile_x: int, tile_y: int) -> bool:
        """Own half only, widening into a lane whose tower has fallen.

        Mirrors ``GameBoard.is_placeable`` in effect: a side may deploy on its
        own half, plus the opposing quadrant behind any tower it destroyed.
        """
        if not (0 <= tile_x < ARENA_COLS and 0 <= tile_y < ARENA_ROWS):
            return False
        own_half = tile_y >= arena.FRIENDLY_HALF_START_ROW if friendly else tile_y < arena.FRIENDLY_HALF_START_ROW
        if own_half:
            return True

        left_key = "enemy_left" if friendly else "friendly_left"
        right_key = "enemy_right" if friendly else "friendly_right"
        left_gone = left_key in self.destroyed_towers
        right_gone = right_key in self.destroyed_towers
        on_left = tile_x < ARENA_COLS // 2
        return (left_gone and on_left) or (right_gone and not on_left)

    def deploy(self, friendly: bool, name: str, tile_x: int, tile_y: int) -> bool:
        """Place a card. Returns False if it was not affordable or legal."""
        from ..game.cards import get_card_cost

        if not self.can_deploy(friendly, name, tile_x, tile_y):
            return False
        self.elixir[friendly] -= get_card_cost(name)
        px, py = arena.deploy_position(tile_x, tile_y)

        if name in SPELL_DAMAGE:
            self._cast_spell(friendly, name, px, py)
            return True

        stats = self.unit_stats[name]
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
        stats = self.unit_stats[name]
        x, y = arena.clamp_to_arena(x, y)
        e = Entity(
            uid=next(_uid_counter), name=name, friendly=friendly, x=x, y=y,
            hp=stats.hp, max_hp=stats.hp, stats=stats,
            deploy_remaining=stats.deploy_time,
            lifetime_remaining=stats.lifetime,
            spawn_cooldown=stats.spawn_period,
        )
        self.entities[e.uid] = e
        return e

    def _cast_spell(self, friendly: bool, name: str, x: float, y: float) -> None:
        damage, radius, building_mult = SPELL_DAMAGE[name]
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

    def _on_death(self, e: Entity) -> None:
        e.hp = 0.0
        if e.is_tower:
            self._on_tower_destroyed(e)
            return
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

    def _acquire_target(self, e: Entity) -> Entity | None:
        """Nearest legal enemy within aggro range, else the goal tower.

        Buildings-only attackers skip the aggro step entirely - that is what
        makes a Giant walk past a Musketeer shooting it, and it is one of the
        few behaviours worth getting exactly right, since whole strategies
        are built on it.
        """
        enemies = [
            o for o in self.entities.values()
            if o.alive and o.friendly != e.friendly and can_attack(e, o)
        ]
        if not enemies:
            return None

        if e.stats.targets is Target.BUILDINGS:
            return min(enemies, key=lambda o: arena.distance(e.x, e.y, o.x, o.y))

        in_aggro = [
            o for o in enemies
            if not o.is_tower
            and arena.distance(e.x, e.y, o.x, o.y) <= e.stats.aggro_range
        ]
        if in_aggro:
            return min(in_aggro, key=lambda o: arena.distance(e.x, e.y, o.x, o.y))

        towers = [o for o in enemies if o.is_tower]
        if towers:
            return min(towers, key=lambda o: arena.distance(e.x, e.y, o.x, o.y))
        return min(enemies, key=lambda o: arena.distance(e.x, e.y, o.x, o.y))

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
                self._on_death(e)
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

        target = self.entities.get(e.target_uid) if e.target_uid else None
        if target is None or not target.alive or not can_attack(e, target):
            target = self._acquire_target(e)
            e.target_uid = target.uid if target else None
        if target is None:
            return

        reach = e.stats.attack_range + target.radius
        dist = arena.distance(e.x, e.y, target.x, target.y)

        if dist <= reach:
            if e.attack_cooldown <= 0.0 and e.stats.damage > 0.0:
                self._attack(e, target)
            return

        if e.stats.speed > 0.0:
            self._move_toward(e, target, dt)
        else:
            # Immobile and out of reach: drop the target so it re-acquires
            # something it can actually hit next tick.
            e.target_uid = None

    def _attack(self, e: Entity, target: Entity) -> None:
        e.attack_cooldown = e.stats.hit_speed
        self._damage(target, e.stats.damage)
        if e.stats.splash_radius > 0.0:
            for other in list(self.entities.values()):
                if other.uid == target.uid or not other.alive:
                    continue
                if other.friendly == e.friendly or not can_attack(e, other):
                    continue
                if arena.distance(target.x, target.y, other.x, other.y) <= e.stats.splash_radius:
                    self._damage(other, e.stats.damage)

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
        nx = e.x + dx / dist * step
        ny = e.y + dy / dist * step

        if not e.stats.flying and arena.blocks_ground(nx, ny):
            # Walked into water: slide along the bank toward the bridge
            # instead of stopping dead against it.
            nx = e.x + (arena.nearest_bridge_x(e.x) - e.x) * min(1.0, step)
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
