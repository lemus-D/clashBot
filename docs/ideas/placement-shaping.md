# Idea: teaching the policy WHERE to place

Status: **thought, not a plan.** Nothing here is built. Recorded 2026-08-24
while looking at `run1`, the checkpoint that plateaued at ~33% headline
against a 24% random baseline.

The problem it responds to: the policy has learned *when* to play a card and
not *where*. Entropy on the tile head moved 5.72 -> 5.61 over 206 updates
against a maximum of ~7.05, i.e. the 144-way tile choice is still close to a
coin toss after 844k steps.

## Darwin's idea: a spatial gradient on the reward

Reward tiles by *area* rather than individually — a smooth gradient that pays
more near good placements (at the bridge, behind the king tower) and falls off
with distance, so the policy is gently pushed toward sensible regions instead
of having to learn 144 unrelated numbers.

The gradient should be **conditioned on where the enemy's cards are**, not
fixed. The weak version worth starting from: when there are no enemy troops on
the board at all, placing on your own side is almost always right, and that
much could be guided without much risk of teaching something false.

**Darwin's own worry, recorded because it is the right worry:** the policy
could overfit to the gradient and place there regardless of what the enemy is
doing — learning the shaping term instead of the game. A reward that says
"behind the king tower is good" will be farmed by a policy that puts
everything behind the king tower.

### Notes on making that safe

- **Potential-based shaping** is the standard defence. If the extra reward is
  expressed as `F(s, s') = gamma * PHI(s') - PHI(s)` for some potential
  function `PHI` over states, the optimal policy is provably unchanged — the
  shaping can only speed up credit assignment, not move the target. Any other
  shape (a flat bonus for placing on a good tile) genuinely does change what
  is optimal, which is exactly the overfitting Darwin is describing. This is
  the difference between "help it find the answer sooner" and "give it a
  different answer".
- Shaping is cheap to *remove*. Anneal the coefficient to zero over training
  and the final policy is scored on the real reward, so a gradient that was
  useful early cannot distort the endgame.
- The enemy-conditioned part is where the theory gets thin: `PHI` has to be a
  function of the state, and "where the enemy's units are" is in the state, so
  it is admissible — but hand-authoring a good `PHI` over enemy positions is
  most of the difficulty of playing the game. The no-enemies-on-board case is
  attractive precisely because it is the one slice where the right answer is
  unambiguous.

### The architectural alternative, which is cheaper

Before shaping: the tile head is a `Linear(256, 144)` over a flat trunk. It
has no idea tile 37 is adjacent to tile 38, so "near my tower" cannot
generalise — it has to be learned 144 separate times. A conv over the 9x16
grid builds that adjacency in structurally, and needs no reward change and no
hand-authored notion of good placement. **This is the cheaper experiment and
should be tried first** — if the policy still cannot place after it, the
shaping idea is much better motivated.

## The counter-idea: opponents that punish bad placement

Rather than rewarding good placement directly, make bad placement *lose*.
Give the scripted opponents the ability to capitalise on a misplaced card —
then the existing win/loss signal already contains "that was the wrong tile",
and nothing has to be hand-specified about where cards belong.

This is the more attractive framing of the two, for three reasons:

1. It cannot be farmed. There is no shaping term to exploit — the policy has
   to actually stop misplacing cards.
2. It raises the ceiling rather than the floor. The current opponents are the
   only absolute yardstick this project has, and they are beatable by bad
   placement, which is *why* placement carries no gradient right now.
3. It reuses the `STYLES` table. A punisher is plausibly a table entry plus a
   targeting rule, not a new system.

**The cost, and it is real:** the scripted opponents are FROZEN on purpose.
They are the only stable measuring stick — self-play sits at ~50% by
construction and measures nothing. Changing them invalidates every number in
this repo, including the 24% random baseline and the 33% headline. So a
punisher should be **added as a new archetype**, not retrofitted onto the four
that exist, and the whole pool re-baselined against RandomPolicy afterwards.

## Kelvin's question: would harder opponents be too hard to learn against?

Real risk, and it has a name: if the policy never wins, every episode returns
the same terminal reward, the advantage estimates carry no signal about which
action was better, and it learns nothing. That is not a slow-learning regime,
it is a zero-gradient one.

Three things make it less likely to bite here than it sounds:

- The reward is **not** win/loss only any more. Tower damage and elixir trades
  pay out during the match, so a policy that loses every game still gets a
  gradient telling it which losses were less bad. This is the same sparsity
  fix that unblocked learning the first time.
- The opponent pool is **sampled per episode**, not graduated through. Keeping
  `bigspender` in the mix means there is always a rung the policy can win on
  while a punisher is still beating it.
- The pool is the curriculum. Adding a hard opponent alongside easy ones is
  automatically incremental — no staging mechanism required.

**On incremental-vs-compute:** the chess/Go precedent cuts both ways and it is
worth being precise about it. AlphaZero used no hand-built curriculum and no
scripted opponents at all — self-play generates the curriculum for free,
because the opponent is always exactly as good as you are. But it also used a
tree search at both training and play time, which is doing a great deal of the
credit assignment that PPO here has to get from the reward alone. "Throw
compute at it" worked there in the presence of search, which is not the
situation here.

The honest position: **incremental is the safer default and costs little**,
because sampling a mixed pool is already incremental. Whether more compute
alone would clear the plateau is untested — `run1` was killed at 206 of 488
updates, so nobody has yet run this to completion even once. That is a cheaper
question to answer than either idea on this page.

## Measured while writing this (2026-08-24)

Watching `models/run1.pt` play revealed a concrete defect that is likely
holding placement back independently of everything above:

- **The observation's `playable_mask` and the simulator's `is_placeable`
  disagree.** Over 8 episodes, **300 of 503 placement attempts (60%) were
  rejected**, every one of them a case where the mask the policy was handed
  said the tile was legal and the env then refused it.
- All 300 were the **same tile**, `(8, 3)`, on the enemy half. Because
  evaluation runs the policy deterministically (argmax), once its top-ranked
  tile is one the mask permits and the env rejects, it retries that tile every
  step for the rest of the match, paying `invalid_action` each time.
- The two rules genuinely differ in the source. `GameBoard.is_placeable`
  forbids the bridge row (`tile_y == 7`) for ground troops;
  `Simulation.is_placeable` has no row-7 rule at all. That divergence runs the
  *other* way and is separate from the `(8, 3)` case, which is on the enemy
  half and so turns on the tower-destroyed / king-active unlock condition —
  the mask is built from OBSERVED tower state (noisy, debounced, can be stale)
  while the sim checks ground truth.
- Inferred, not confirmed: a stale or mis-debounced enemy-tower reading opens
  the enemy quadrant in the mask while the sim still considers it closed. The
  exact trigger has not been isolated.

This matters for this page because a policy whose placements are rejected 60%
of the time is being taught almost nothing about placement — the tile head's
near-uniformity may be a *consequence* of this rather than an architectural
limitation. **Fix the mask disagreement and re-measure before investing in
either the shaping gradient or the punisher opponent.**
