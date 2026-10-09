# Games: decomposition, part by part

Before this port the package was 241 lines of crisp TypeScript: three world classes (economy, combat, NPCs) with
every rule in code, and one inline `nl` call per actor policy. The rules were crisp; only the choice of a move was
natural language. Now the rules are natural language too, and crisp code checks what the stages produce.

Every part of every world, with a decision:

- **fn**: its own natural-language function with a typed contract.
- **inline**: one instruction inside its caller.
- **implicit**: left to the model (rare).
- **crisp**: a TypeScript helper in a callable folder (trivially deterministic plumbing, or the check).
- **pluggable**: one interface, two implementations (natural language and crisp), selected by `Settings`.
- **host**: the outside world, the CLI, exact durability.

The executors are small, fast models, so each function is one task a small model can finish, its algorithm is
spelled as numbered steps over named data, and exact arithmetic is done in eval.

## Policy

- **The state model, after Elm and Clojure.**
  - A world is a value (`EconomyState`, `CombatState`, `NpcState`). A turn is a function of it.
  - *Decide on a snapshot*: observe, choose, validate, settle or resolve are asynchronous stages that read the
    snapshot and produce **effects as data** (`EconomyEffects`, `CombatEffects`, `NpcEffects`): the submissions, the
    order, the settlement or resolution, the notes.
  - *Apply a pure, bounded commit*: `commit` takes the state and the effects, checks them, and returns the next
    state with the events, or the state unchanged with the problem. It has no model call, no loop without a bound,
    and no clock. The commit line is the only place a state changes.
  - *Timers are state*: the fighter's `cooldown` is a counter in the state, decremented by the round's upkeep.
    The NPC world's `nextEvent` and `applied` are state too, not hidden host counters.
  - *Derived values form a DAG*: observation from state; validation from state and observation; order from the
    seed and the tick; settlement from balances, order and intents; damage from hits; narration from events. Nothing
    reads a value computed later.
  - *Approximately Turing-incomplete*: no `while`. Every sequence is `for...of` over a list. A stage that failed
    its commit is retried once with the problem, and then the turn reports `ok: false`; there is no open loop.
- **Natural language: the rules.** That is what the old classes encoded:
  - turn structure (who acts, in which phase, what a failed turn does);
  - intent, tactic and plan validation;
  - settlement, with its rejection conditions;
  - combat resolution: movement, strikes, wounds, cooldown timers;
  - NPC memory update and planning;
  - narration.
- **Crisp: only what checks or is plumbing.**
  - observation (the information boundary: an actor sees exactly its own fields and the public ones);
  - the seeded order (SHA-256, a random source: a model cannot hash);
  - ledger entries from an outcome, and applying entries to balances;
  - the commits, which verify what the stages produced;
  - the initial-state constructors and the host session (compare-and-set of the state).
- **Crisp verification of the stages, not a crisp core.**
  - *Conservation*: money and each good sum to the initial totals after every settlement, balances never go
    negative, and the entries are exactly the traded outcomes' two entries each.
  - *Determinism*: the order is recomputed from seed and tick; a rejected trade is re-derived as infeasible at that
    point of the settlement, a traded one as feasible at the offer's price. Two runs from the same state and
    intents give the same settlement whatever order the intents arrived in.
  - *Combat*: every fighter stays in the arena, steps at most one cell, only the living with a tactic move, health
    equals before minus the hits, every hit is legal on the state it was judged on (attacker alive and off cooldown,
    target named by the tactic and alive, within one cell after movement, damage 2 or 1 against a guard, one hit per
    attacker), and a cooldown starts at a strike and otherwise only counts down.
  - *Soundness, not completeness.* The checks bound what a stage may do, not what it must do. A policy may reject
    a trade or miss a hit that the physics allows; it cannot create money, goods, health or reach. Rules that
    tighten the game in natural language stay possible without touching the checks.
  - *NPCs*: an event is acted on once (`basis`, `applied`, `event-N`), and a give removes a unit that exists.
- **Pluggable hot paths.** Policy is natural language also where it runs for every actor of every turn. Each such
  part has one interface and two implementations, crisp and natural language, selected by `Settings`
  (`Engine = 'nl' | 'crisp'`). Both are held to the same commit checks.
  - `validate`: intent, tactic and plan legality (per actor per turn);
  - `settle`: the economy's trades (per tick);
  - `resolve`: the combat round (per round);
  - `remember`: NPC notes (per event);
  - `narrate`: the turn's story (per turn).

  The actors' own choice (`choose`, `tactic`, `plan`) is the policy being played: natural language only. The
  mechanisms (commit, order, storage, the session) stay host code.
- **Messages teach positively.** Types and doc comments in `types.ts` state each wanted property. A validation
  verdict and a commit problem say which property failed; the retry carries that sentence as `problem`. No
  instruction lists wrong forms.

## Turn structure (games.nl)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Dispatch on the scene's kind | inline | `games` | A three-way choice of the world's turn. |
| The world's turn (observe, choose, validate, settle or resolve, commit) | fn, per world | `games/economy`, `games/combat`, `games/npc` | Each world is its own sequence with its own data. |
| Narration of the turn from its events | pluggable | `games/narrate` | Per turn, optional; crisp is a plain listing. |
| Return the report with the next state | inline | `games` | Assembly. |
| Session: hold the state, step, compare-and-set | host | `index.ts` | Durability and exactly-once belong to the host. |
| Initial states, from lists of merchants, fighters, NPCs | host | `index.ts` | Input boundary of the outside world. |
| CLI demo scenes | host | `main.ts` | The CLI. |

## Economy

| Part | Decision | Unit | Why |
|---|---|---|---|
| Observation: own cash and goods, public offers of the others | crisp | `economy/observe.ts` | The information boundary. A model must not decide what is secret. |
| Choose buy or pass from an observation | fn | `economy/choose` | The merchant's policy. |
| Intent validation: seller is another merchant, good is a name, quantity a positive integer | pluggable | `economy/validate` (`validate/judge.nl`, `validate/rules.ts`) | Per intent. Stock and cash are settlement's concern. |
| An invalid intent becomes a pass, with the reason in the log | inline | `economy` | Orchestration. The old code threw; a rejected intent now costs only that merchant its turn. |
| Seeded order of the tick: sort by SHA-256 of `seed:tick:actor` | crisp | `economy/order.ts` | A random source; models cannot hash. The commit recomputes it. |
| Settlement of all intents in order against running balances | pluggable | `economy/settle` (`settle/policy.nl`, `settle/reference.ts`) | Per tick. The old `settle`. |
| One intent against the running balances: pass, rejected with reason, or traded at the offer's price | fn | `settle/policy/quote` | The rule of the market. One small decision, run for each intent in order. |
| Entries of a traded outcome; applying entries to balances | crisp | `economy/ledger.ts`, used by `settle/policy` and the commit | Arithmetic over named data; shared so the NL path and the check use the same definition. |
| Retry settlement once with the commit's problem | inline | `economy` | Orchestration. |
| Commit: basis, order, entries well formed, balances never negative, outcomes re-derived, totals equal the initial ones | crisp | `economy/commit.ts` | The check and the atomic state change. |
| Events (`economy.intent`, `economy.settle`) | crisp | `economy/commit.ts` | Data derived from the effects. |

## Combat

| Part | Decision | Unit | Why |
|---|---|---|---|
| Observation: self, others' cells and health | crisp | `combat/observe.ts` | Information boundary. |
| The fighter's tactic | fn | `combat/tactic` | The fighter's policy. |
| Tactic validation: move and action in their sets, an attack names another living fighter | pluggable | `combat/validate` | Per tactic. Old code allowed attacking oneself; a self-attack is now invalid, and the reason says so. |
| An invalid tactic becomes stay and rest | inline | `combat` | Orchestration. |
| Dead fighters submit nothing | inline | `combat` | Turn structure. |
| Resolution of a round | pluggable | `combat/resolve` (`resolve/policy.nl`, `resolve/reference.ts`) | Per round. |
| Movement, simultaneous: each living submitter steps one cell, clamped to the arena | fn | `resolve/policy/move` | Phase 1, its own rule. |
| Strikes: attacker alive, off cooldown, attacking a living target within one cell after movement; damage 2, or 1 against a guard | fn | `resolve/policy/strike` | Phase 2. Reads positions after movement, and health and cooldown before the round. |
| Wounds: health falls by the damage taken, to a floor of 0 | fn | `resolve/policy/wound` | Phase 3, applied after all strikes so they are simultaneous. |
| Cooldown timers: a fighter that struck gets 1, others count down to 0 | fn | `resolve/policy/recover` | Phase 4, the timers. |
| Damage per target from the hits | inline | `resolve/policy` | A sum. |
| Commit: invariants of a round (see Policy) | crisp | `combat/commit.ts` | The check and the atomic state change. |

## NPCs

| Part | Decision | Unit | Why |
|---|---|---|---|
| Observation: inventory, memory including the new event, commitments | crisp | `npc/observe.ts` | The information boundary. The event id is `event-N` from the state's counter. |
| Memory update: notes that summarize the event with the memory | pluggable | `npc/remember` (`remember/notes.nl`; crisp: no notes) | Per event. Notes cite the entries they rest on. The event itself is always stored verbatim by the commit. |
| Plan: say, and one of none, give, promise | fn | `npc/plan` | The NPC's policy, from its memory and commitments. |
| Plan validation: item available, targets named, promise concrete | pluggable | `npc/validate` | Per plan. |
| An invalid plan keeps the speech and takes no action | inline | `npc` | Orchestration. |
| Commit: basis, event not yet applied, id is next, give removes a unit that exists, promise cites the event | crisp | `npc/commit.ts` | The check, the exactly-once line, the atomic state change. |
| Event (`npc.act`) | crisp | `npc/commit.ts` | Data derived from the effects. |

## Narration

| Part | Decision | Unit | Why |
|---|---|---|---|
| Story of a turn from its events (what traded, who hit whom, what an NPC said and did) | pluggable | `games/narrate` (`narrate/story.nl`, `narrate/plain.ts`) | Read-only; never changes state. |

## Refinements

Constraints that should become refinement types (`Is<T, "predicate">`, plans/REFINEMENT_TYPES.md). They are not
written in the code, which carries them as doc comments and crisp checks; each is listed with its slot.

| Slot | Proposed type |
|---|---|
| `Merchant.id`, `Fighter.id`, `NpcActor.id`, `Submission.actor`, `TradeIntent.seller`, `TradeIntent.good`, `CombatPlan.target`, `NpcPlan.item`, `NpcPlan.target` | `Is<string, "a name: a letter followed by letters, digits, '_' or '-'">` |
| `Merchant.cash`, `Merchant.goods[*]`, `Merchant.offers[*]`, `Entry.amount` bound, `TradeOutcome.total` | `Is<number, "a non-negative safe integer">` |
| `TradeIntent.quantity` when kind is buy, `Entry.amount` | `Is<number, "a positive safe integer">` |
| `EconomyState.merchants` | `Is<Merchant[], "at least two merchants with distinct ids">` |
| `EconomyState.initial` | `Is<Totals, "the sum of the merchants' cash and goods at the start">` |
| `Settlement` | `Is<Settlement, "entries are the traded outcomes' goods-then-cash pairs in order">` |
| `EconomyEffects.order` | `Is<string[], "the submitters sorted by SHA-256 of seed:tick:actor">` |
| `Settlement` (as the commit's argument) | `Is<Settlement, "conserves each good and the money, and no balance goes below zero">` |
| `TradeOutcome` with status traded | `Is<TradeOutcome, "total equals quantity times the seller's offered price">` |
| `Fighter.x` | `Is<number, "an integer from 0 to the arena's width minus 1">` |
| `Fighter.hp` | `Is<number, "an integer from 0 to the fighter's starting health">` |
| `Fighter.cooldown` | `Is<number, "0 or 1">` |
| `CombatState.fighters` | `Is<Fighter[], "at least two fighters with distinct ids">` |
| `CombatPlan.target` when action is attack | `Is<string, "the id of a living fighter other than the actor">` |
| `Hit.amount` | `Is<number, "2, or 1 when the target guarded">` |
| `CombatResolution` | `Is<CombatResolution, "each fighter moved at most one cell; health fell by exactly the damage taken; each eligible attacker landed exactly one hit">` |
| `NpcPlan.item` when action is give | `Is<string, "an item the NPC holds at least one of">` |
| `NpcPlan.detail` when action is promise | `Is<string, "a concrete, non-empty commitment">` |
| `Note.about` | `Is<string[], "ids of entries in the NPC's memory">` |
| `Commitment.evidence_id` | `Is<string, "the id of the event that prompted the promise">` |
| `NpcEffects.event_id` | `Is<string, "the id the state would assign next, not yet applied">` |
| `NpcPlan.say` | `Is<string, "a reply in the NPC's voice that claims no knowledge outside its memory">` |
| `Settings` entries | `Is<Engine, "an implementation that exists for the part">` |

## Limitations met while porting

- **Callable-folder TypeScript cannot import node modules**, so the SHA-256 of the seeded order is written out in
  `games/economy/order.ts` (about 40 lines, checked against `node:crypto` in the tests). A `random` service would
  replace it.
- **`natlang check` does not apply the finite-iteration policy to callable-folder TypeScript**: `for (... of
  array.entries())` passed `check` and was rejected only when the turn ran (ts-host/src/native/runtime.ts, the eval
  policy message "iterates an array, string, Map or Set"). The commits use counted loops.
- **Local type inference from an initializer can pick the wrong type.** `const picks = await Promise.all(ids.map(async
  id => ({ actor: id, intent: await choose(...) })))` was given the type `TradeIntent[]` (the callee's return) and the
  binding was rejected with `type-mismatch`; an annotation (`const picks: Submission[]`) fixes it
  (ts-host/src/native/runtime.ts:1982, `scopeInitializerType`). The stage instructions name their data types, but a
  small model that does not annotate hits this.
- **Types: no intersections or index signatures** in `types.ts` (`A & B` and `{ [k: string]: unknown }` fail
  `callable-scope` parsing), so `TurnReport` repeats the fields of `Turn` and `GameEvent` is `Record<string, unknown>`.
- **Names that collide with function properties** (`apply`) are rejected in callable folders; the ledger uses `post`.
- **A helper two items need** (`economy/ledger.ts`) is both a child of `economy` (so `economy` sees it as a callable)
  and reached by `settle/policy` through `uses`. It cannot be hidden from the economy's eval scope.
- **No refinement types yet**, so all constraints in "Refinements" are doc comments plus the commit checks.
- **No shadow comparison** (NL against crisp on the same input, disagreements recorded) between the two
  implementations of a pluggable part; plans/REFINEMENT_TYPES.md describes it for refinements and it would apply
  here. The tests compare them on fixed scenarios.
