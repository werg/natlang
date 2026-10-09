# Games: worlds whose rules are natural language

A market that settles in a seeded order, a duel arena that resolves simultaneously, and NPCs that act from their own
memory. The rules are natural-language stages; crisp code only checks what the stages produce. The part-by-part
design is in [DECOMPOSITION.md](DECOMPOSITION.md).

```
games.nl                         one turn of a scene: dispatch, narrate, report
  games/economy.nl               observe, choose, validate, order, settle, commit
  games/combat.nl                observe, tactic, validate, resolve, commit
  games/npc.nl                   observe, remember, respond, validate, commit
  games/narrate                  the story of the turn
```

## State model

- A world is a value. A turn **decides on a snapshot** (asynchronous stages, model calls) and then **commits once**:
  a pure, bounded function from state and effects to the next state and events, or to the problem.
- Effects are data (`EconomyEffects`, `CombatEffects`, `NpcEffects`). Timers are state (a fighter's `cooldown`, the
  village's `nextEvent` and `applied`). Derived values form a DAG: observation, validation, order, settlement or
  resolution, narration.
- A stage refused by the commit is run once more with the problem; a second refusal leaves the world as it was and the
  turn reports `ok: false`.

## What the commits check

- **Economy**: the settlement order is the seeded one (SHA-256 of `seed:tick:actor`); the entries are exactly the
  traded outcomes' goods and cash pairs; a trade is at the seller's offered price; no balance goes below zero; money
  and every good keep their initial totals.
- **Combat**: movement of at most one cell inside the arena; health falls by exactly the damage taken; every hit is
  legal (alive, off cooldown, in reach after movement, 2 damage or 1 against a guard, one per attacker); cooldowns start
  at a strike and otherwise count down.
- **NPCs**: an event is acted on once (version, `event-N`, `applied`); a give removes an item the NPC holds; notes cite
  entries of its memory.

## Pluggable parts

`Settings` selects `'nl'` or `'crisp'` for each of `validate`, `settle`, `resolve`, `remember` and `narrate`. Both
implementations sit behind one interface (a TypeScript module that imports the natural-language function and the crisp
reference) and are held to the same commit. `defaultSettings` is all natural language; `crispSettings` leaves only the
actors' own choices (`choose`, `tactic`, `respond`) to a model.

## Use

```ts
import { Session, createEconomy, defaultSettings } from './index.js';
const session = new Session(createEconomy(merchants, { seed: 33 }), defaultSettings, runtimeBoundPlayTurn);
const report = await session.turn();   // report.state, report.events, report.narration
```

`natlang run applications/games -- economy|combat|village [--turns N] [--crisp]` plays a demo world.

Tests: `ts-host/test/games.test.mjs` (scripted stages; the commits are also loaded straight from source).
