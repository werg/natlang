# P08 — Economic, combat and NPC game prototypes

Status: three headless prototypes in `codebases/game_economy/`,
`codebases/game_combat/`, `codebases/game_npc/` and
`applications/game_worlds.mjs`. [Shared capabilities](README.md).

Natlang chooses merchant trades, fighter tactics and NPC dialogue/actions.
The exact hosts settle seeded simultaneous trade intents with conservation,
resolve movement and damage at round boundaries, and guard NPC inventory and
promise provenance. Actor observations expose only assigned information.
Integration tests execute all three natlang policy entry points, check legal
effects, determinism under reordered trade submissions, stale combat plans,
and forged/replayed NPC observations.

These are local headless worlds. Graphical rendering, deeper economic goals,
real-time tactic deadlines, interacting NPC quests, parallel actor policy
execution, and live-model quality evaluation remain product gates. No game
engine primitive was added to the natlang core.

## Natlang prerequisites

C0 provides typed observations/actions and policy functions. C2 separates world randomness from model-policy randomness. C5 supplies the game event stream; finite rounds can use current Fold first. C6 later batches independent policies. C1 exposes native world/rendering objects where helpful. No actor type, shared-memory mutation primitive or in-episode interrupt is required.

## Programme and typed boundary

Economic game: `decide_trade.nl(observation, goals) -> TradeIntent`; world state contains exact quantities, integer money and offers. Natlang chooses interpretation/strategy; crisp code validates and settles trades.

Combat: `choose_tactic.nl(perception, memory) -> CombatPlan`; exact simulation owns movement, collision, cooldowns and legal actions. Natlang produces short plans at decision boundaries, not individual physics frames.

NPCs: `react.nl(event, memory, goals) -> ResponsePlan`; natlang chooses dialogue/commitments/actions. Exact helpers apply valid inventory/quest changes. Extend the existing shopkeeper example first.

Only bounded observations, plans and state summaries cross the tree boundary. Native world objects, render assets and memory indexes remain in eval. An actor observes the information assigned by the game rules; direct host sharing must not accidentally expose all private actor data to every policy invocation.

## Environment and reduction

The host creates immutable observation snapshots and a stream of tick/player/world events. One Fold owns authoritative world transitions. Policy calls can be mapped over actors with isolated read snapshots; exact resolution applies simultaneous intents under declared rules. Seed streams are keyed by actor/tick and purpose, not job completion order.

A host-backed `ts` engine can access rendering/world APIs directly. Policy-facing bindings should match the intended game information contract. Concurrent code cannot freely mutate shared authoritative state before resolution. Long policy decisions can become pending host work whose result arrives later; the game validates the observation revision and applies a declared fallback on deadline. An event does not rewrite a currently running policy lambda.

## Delivery and checks

1. Economy: three merchants/two goods over fixed rounds. Gate: conservation and legal settlement; strategic quality scored separately.
2. Combat: turn-based arena, then lower-frequency tactics within a real-time host loop. Gate: every accepted plan is legal; stale/late results cannot teleport the world or stall rendering.
3. NPC: persistent shopkeeper, then a small interacting group. Gate: actions respect inventory/commitments and memory evidence is attributable.
4. Add independent parallel policy execution only after matched serial runs pass the intended contract.

Test same-name entities, conflicting goals, simultaneous actions, missing observations, interruptions represented as next events and exhausted policy budgets.

## Trace, training and portability

Capture observations, policy outputs, logical seeds and exact world transitions. Preserve fallback actions and late decisions in the trace. Teacher examples need legal-action oracles and separate behavior rubrics. Hold out world layouts, goals and actor compositions. A graphical host and a headless simulator can share policy code and trace format without sharing native rendering objects.
