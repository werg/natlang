---
description: One tick of a market of merchants. Every merchant chooses from its own observation, the intents are validated, settled in the seeded order, and committed once.
args:
  state: EconomyState
  settings: Settings
returns: Turn
---
Play one tick of the economy in state with the stages in your folder. The tick decides on the snapshot state and then
commits once; nothing changes before the commit.

1. ids = the ids of state.merchants. For every id at the same time: observation = observe(state, id), then intent =
   choose(observation). A submission is { actor: id, intent }.
2. Validate every submission at the same time: verdict = validate(ids, submission, settings). When verdict.ok is
   false, replace that submission's intent by { kind: "pass" } and add the log line "<actor>: <verdict.reason>".
3. ordered = the submissions arranged by order(state.seed, state.tick, actors), where actors are their actor ids.
   order is the list of actors it returned.
4. settlement = settle(state, ordered, settings, ""). effects = { basis: state.tick, order, submissions: ordered,
   settlement }. committed = commit(state, effects).
5. When committed.ok is false, settle once more with problem committed.problem as the last argument of settle, build
   the effects from that settlement and commit again. When that fails too, return { ok: false, state, events: [], log,
   problem } with the problem of the last commit: the tick does not happen and state stays as it was.
6. Otherwise return { ok: true, state: committed.state, events: committed.events, log, problem: "" }.

log holds the lines of step 2 and, last, one line "settled N trades of M intents" with the counts of settlement.outcomes.
