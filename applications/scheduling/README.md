# Scheduling in natural language

A day has working windows, tasks (durations, bounds, dependencies) and fixed commitments. A request in plain words
("add a 45 minute workout, review not before 11, I'm at the dentist 9:00-9:30, draft early if you can") is turned
into a committed plan by a scheduler written in natural language. Crisp code is the exact verifier and the
revisioned commit. [DECOMPOSITION.md](DECOMPOSITION.md) records the decision for every part.

```
scheduler.nl                  the planner: reads, extracts, builds, repairs, ranks, explains
  scheduler/readTasks.nl        new tasks the request introduces
  scheduler/readLimits.nl       hard limits on tasks, and commitments
  scheduler/readPreferences.nl  weighted wishes
  scheduler/domains.nl          constraint extraction: where each task can go (busy time, free time, bounds, slot grid)
  scheduler/order.ts            dependency order, or the cycle (crisp)
  scheduler/enumerate.ts        PLUGGABLE: complete feasible schedules, exact (crisp) or natural language
    enumerate/construct.nl        depth-first construction over the domains
  scheduler/repair.nl           fixes a near miss from the verifier's violations
  scheduler/assess.nl           one wish against one schedule: met, partly, missed (a decision)
  scheduler/tiebreak.nl         which of two equal schedules serves the request better (a decision)
  scheduler/explain.nl          why this plan
  scheduler/diagnose.nl         why no plan exists, and what to relax
workspace.ts                  Problem (exact verifier, exact enumeration), ScheduleWorkspace (revision, commit)
calendar.ts                   the `calendar` service the stages call: check, exact, describe, implementation
index.ts                      plan(): snapshot, run the scheduler, commit conditionally
```

## How a request runs

1. `plan` freezes the day (a `Problem`) and reads its revision. All times the stages see are minutes after the day's
   origin; a clock-time rendering comes from `calendar.describe`.
2. `scheduler.nl` runs against the frozen day. Nothing it does can change the workspace.
3. The result is a `Proposal`: `chosen`, `infeasible` or `unclear` (with the questions to ask).
4. For `chosen`, `workspace.commitPlan(placements, revision, hard)` re-checks everything with the exact verifier. It
   commits only if the revision is unchanged (otherwise `stale`) and no constraint is broken (otherwise `rejected`,
   with the violations named). The request's new tasks and commitments join the day in the same revision.

```ts
const result = await plan(workspace, 'Add a workout, review not before 11, draft early', {
  run: (fn, options) => runtime.run(fn, options),   // supplies the calendar service
  enumeration: 'natural-language',                  // or 'crisp' (the default)
  candidates: 16,
});
// result.status: committed | stale | rejected | infeasible | unclear; result.explanation is for the user.
```

## Pluggable enumeration

Enumeration is the hot path. `enumeration: 'crisp'` runs the workspace's depth-first search. `'natural-language'` runs
`enumerate/construct.nl`, which builds the same schedules from the domains. Either way every candidate is verified
by `calendar.check`, near misses are repaired (at most 3 rounds), and an empty result is confirmed by the exact search
before a request is called infeasible.

## Tests

`ts-host/test/scheduling.test.mjs` scripts each stage with the code a model would write and checks the host around it:
the verifier's violations, stale and rejected commits, both enumerations, repair, unclear and infeasible requests.
No model is loaded.
