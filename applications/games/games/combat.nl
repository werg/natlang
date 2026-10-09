---
description: One round of a duel arena. Every living fighter chooses a tactic from what it sees, the tactics are validated, resolved all at once, and committed once.
args:
  state: CombatState
  settings: Settings
returns: Turn
---
Play one round of the arena in state with the stages in your folder. The round decides on the snapshot state and then
commits once; nothing changes before the commit.

1. For every fighter of state.fighters whose hp is above 0, at the same time: observation = observe(state, fighter.id),
   then plan = tactic(observation). A submission is { actor: fighter.id, plan }. A fighter with hp 0 submits nothing.
2. Validate every submission at the same time: verdict = validate(state, submission, settings). When verdict.ok is
   false, replace that submission's plan by { move: "stay", action: "rest" } and add the log line "<actor>:
   <verdict.reason>".
3. resolution = resolve(state, submissions, settings, ""). effects = { basis: state.round, submissions, resolution }.
   committed = commit(state, effects).
4. When committed.ok is false, resolve once more with committed.problem as the last argument of resolve, build the
   effects from that resolution and commit again. When that fails too, return { ok: false, state, events: [], log,
   problem } with the problem of the last commit: the round does not happen and state stays as it was.
5. Otherwise return { ok: true, state: committed.state, events: committed.events, log, problem: "" }.

log holds the lines of step 2 and, last, one line "round R: H hits" with the new round number and the number of
resolution.hits.
