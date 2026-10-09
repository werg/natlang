# Scheduling: decomposition, part by part

Every part of the scheduler, with a decision:

- **fn**: its own natural-language function with a typed contract.
- **inline**: instructions inside its caller.
- **implicit**: left to the model.
- **crisp**: a TypeScript helper (callable-folder code, or host code).
- **service**: the outside world, or the exact verifier, reached through the `calendar` service.
- **pluggable**: one interface, a crisp and a natural-language implementation, a setting selects.

The executors are small models, so instructions spell algorithms out as numbered steps over named data structures.

## Policy

- **Natural language: the scheduler.** Everything a scheduler does between a request and a plan:
  - reading the request: new tasks, hard limits, fixed commitments, soft preferences;
  - extracting constraints: free time per task, dependencies;
  - building candidate schedules, and repairing near misses;
  - judging preferences, choosing among valid candidates, explaining the choice or the impossibility.
- **Crisp: the verifier and the commit.** `Problem.check` decides exactly whether a set of placements satisfies every
  hard constraint, and says which constraint fails, in words that name the fix. `commitPlan` applies a verified plan
  under an expected revision. Nothing natural language produces reaches the plan without passing both.
- **Decide on a snapshot, then commit.** A request runs against a frozen `Problem` (the `calendar` service). The
  commit compares the revision it read. A calendar block observed meanwhile makes the plan `stale`. It is never
  merged into a decision made on older facts.
- **Effects are data.** Hard requirements read from the request (new tasks, commitments) are a value, `Hard`. They
  enter the workspace only at the commit, as one revision. The event log (`drainEvents`) is the record.
- **Derived values form a DAG** (one run of `scheduler.nl`):

  ```
  request, view ─▶ readTasks ─▶ readLimits ─┐
                                  readPreferences ┤  (the last two run together)
                                                  ▼
                       view + hard ─▶ domains ─▶ order ─▶ enumerate (pluggable) ─▶ offers
                                                                  offers ─▶ check ─▶ valid / violations
                                                         violations ─▶ repair ─▶ check ─▶ ...
                          valid offers × preferences ─▶ assess ─▶ totals ─▶ tiebreak ─▶ explain
                          nothing valid ─▶ exact enumeration (confirm) ─▶ diagnose
  ```

- **Stay close to JS; no unbounded loops.** There is no `while`. Repair is a counted loop of at most 3 rounds.
  Enumeration stops at `limit`. Tie-breaking walks the tied candidates once.
- **Times are whole minutes after `view.origin`.** The workspace converts to and from UTC minutes. A model never sees
  an epoch number, and never computes a hash: a plan is sealed by the host.
- **Pluggable hot path: enumeration.** Enumerating feasible candidates runs on every request and is the most
  expensive part. `enumerate.ts` has two implementations: `calendar.exact` (the crisp depth-first search) and
  `enumerate/construct.nl`. `calendar.implementation("enumeration")` selects. The setting is `plan(..., { enumeration })`.
  The natural-language implementation is never trusted for infeasibility: when it yields nothing valid, the exact one is
  asked once, and only an exact "none" makes a request infeasible.

## Request reading

| Part | Decision | Unit | Why |
|---|---|---|---|
| New tasks named in the request (id, minutes, earliest, latest, dependencies) | fn | `scheduler/readTasks` | Its own data (`TaskView`). Defaults (the day's extent) are spelled. |
| Hard limits on tasks ("review not before 11:00", "after the call") and commitments ("I'm out 11:00-11:30") | fn | `scheduler/readLimits` | Hard and soft are different contracts: a limit is checked exactly, a preference is judged. |
| Soft preferences ("draft early", "keep the afternoon free"), each with the tasks it concerns and a weight 1 to 3 | fn | `scheduler/readPreferences` | Judged later by `assess`. Independent of `readLimits`, so they run together. |
| Ambiguities that need the user (an unknown task, two readings of a time) | inline | each reader | Each reader returns `questions`. The driver asks back instead of guessing. |
| Merging the readings into `Requirements` | inline | `scheduler` | Concatenation. Spelled in the driver. |

## Constraint extraction

| Part | Decision | Unit | Why |
|---|---|---|---|
| Busy time: fixed commitments plus the request's commitments, merged into disjoint spans | inline | `scheduler/domains` | The first steps of one algorithm. |
| Free time: windows minus busy time | inline | `scheduler/domains` | Same algorithm. |
| Per task: free time inside `[earliest, latest]` (after limits), slot-aligned, long enough for the duration | fn | `scheduler/domains` | The extraction the owner lists: windows, durations, fixed commitments. Its result, `Domain[]`, is data every later stage reads. |
| Dependencies: `after` of a task joined with the limits' `after` | inline | `scheduler/domains` | Carried into `Domain.after`. |
| Dependency order, cycles, unknown ids | crisp | `scheduler/order.ts` | A topological sort has no decision a scheduler book discusses. It reports a cycle in words, and the driver turns that into a question. |
| Non-overlap | not a stage | `construct` and `Problem.check` | It is the rule enumeration respects and the verifier enforces. |

## Candidates

| Part | Decision | Unit | Why |
|---|---|---|---|
| Enumerate feasible complete schedules up to `limit` | pluggable | `scheduler/enumerate.ts` | The hot path (above). |
| Exact enumeration (depth-first, windows in order, starts ascending) | crisp, service | `calendar.exact` | The reference implementation, and the confirmation of infeasibility. |
| Natural-language enumeration: depth-first assignment in dependency order over the domains | fn | `scheduler/enumerate/construct` | Same algorithm, spelled for a small model. |
| Verify each offer: all hard constraints, all violations named | service | `calendar.check` | The verifier. Exact. |
| Repair a near miss from its violations: keep what is valid, move each violating task to the earliest start in its domain after its dependencies | fn | `scheduler/repair` | Judgment about which task moves. At most 3 rounds, counted. |
| Which near miss to repair (fewest violations) | inline | `scheduler` | One comparison. |
| Confirmation by exact enumeration when nothing valid remains | inline | `scheduler` | Orchestration. |

## Preference ranking and explanation

| Part | Decision | Unit | Why |
|---|---|---|---|
| One preference against one candidate: met, partly or missed | fn, decision | `scheduler/assess` | A scored judgment, run for every pair at once. |
| Candidate text for the judges ("draft: 09:00-09:30") | service | `calendar.describe` | Clock arithmetic and formatting. A small model mis-adds minutes. |
| Weighted total (met 2, partly 1, missed 0, times weight) | inline | `scheduler` | Spelled arithmetic. |
| Tie between equal totals | fn, decision | `scheduler/tiebreak` | A judgment about the request as a whole. |
| Explanation of the chosen plan (what it satisfies, what it gives up, what was assumed) | fn | `scheduler/explain` | Customer-facing text. |
| Diagnosis when infeasible (the conflict, and what to relax) | fn | `scheduler/diagnose` | Its own task. Reads the domains: an empty domain names the task. |

## Mechanism (host)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Time parsing with explicit offsets, integer minutes | crisp | `workspace.ts` | Exact. |
| The workspace: revision, observations of calendar blocks, event log | crisp | `workspace.ts` | State. |
| Exact verifier (`Problem.check`), exact enumeration | crisp | `workspace.ts` | The oracle. |
| Revisioned commit (`commit`, `commitPlan`), including admission of new tasks and commitments | crisp | `workspace.ts` | Exact durability of a decision. |
| The run: snapshot, run `scheduler.nl` with the `calendar` service, commit, attach the explanation | crisp | `index.ts` | Plumbing around the decision. |

## Refinements

Refinement types (`Is<T, "predicate">`, plans/REFINEMENT_TYPES.md). Decisions: (a) adopted with a crisp checker in
`refinements.ts` (no model call), (b) a natural-language judge (proposed only: awaiting live evaluation, not wired),
(c) left to the check that already enforces it exactly (`calendar.check`, the order stage, the commit), (d) not adopted:
the value alone does not show the property. The refined result types are `Checked*` aliases in `types.ts`, named in
the `returns` of the stage that produces the value; the host and the crisp code keep the plain types. `index.ts`
exports the table as `refinements`, for a host that runs the planner in its own runtime (`refinements: { crisp }`).

| Slot | Proposed type | Decision |
|---|---|---|
| `TaskView.id` | `Is<string, "an identifier that starts with a letter and has only letters, digits, underscores and hyphens">` | (a) `CheckedTaskReading`, with no id used twice among the tasks read. |
| `TaskView.minutes` | `Is<number, "a whole number of minutes, at least 1">` | (a) `CheckedTaskReading`. |
| `Span`, `Placement`, `Block` | `Is<Span, "start is a whole number of minutes before end">` | (a) `CheckedLimitReading` (blocks), `CheckedPlacements` (repair), `CheckedDomains` (spans). |
| `Placement.start` | `Is<number, "a multiple of the slot length after the origin">` | (c) The slot is not in the placement; `calendar.check` reports `grid`. |
| `TaskView.after` | `Is<string[], "ids of other tasks, forming no cycle">` | (c) `order` reports the cycle as the ordering's problem; the graph spans several tasks and the day. |
| `TaskReading.tasks` | `Is<TaskView[], "tasks the request introduces, none of them already in the day">` | (d) The day is not in the value. |
| `LimitReading.limits` | `Is<Limit[], "each limit narrows one task and restates a requirement of the request">` | (a) Weakened (`CheckedLimitReading`): each limit names a task, gives a reason and sets notBefore, endBy or after. That it restates the request is (d). |
| `Preference.text` | `Is<string, "the user's wish, restated without times the request does not state">` | (d) The request is not in the value. |
| `Preference.weight` | `Is<number, "1, 2 or 3: 3 for a wish stated as important">` | (a) `CheckedPreferenceReading`: weight 1, 2 or 3, ids numbered p1, p2 in order. Whether 3 was stated as important is (d). |
| `Preference.tasks` | `Is<string[], "ids of tasks in the day, or empty for the whole plan">` | (d) The day is not in the value; the list of ids is shape-checked in `CheckedPreferenceReading`. |
| `Domain.spans` | `Is<Span[], "free of fixed commitments, inside the task's earliest and latest, each long enough for the task, ascending and disjoint">` | (a) Weakened (`CheckedDomains`): ascending, disjoint, valid spans, each at least the task's minutes. Free of commitments and inside the bounds needs the day; `calendar.check` rejects a schedule that breaks them (c). |
| `Offered.options[].placements` | `Is<Placement[], "one placement per task, none overlapping, each after its dependencies">` | (c) The verifier's job, as the row says. |
| `Proposal.placements` | `Is<Placement[], "a complete schedule that calendar.check accepts">` | (c) `calendar.check` before the commit, and the commit again. |
| `Explanation.text` | `Is<string, "plain words for the user, naming each wish that is met and each that is given up, without minute offsets">` | (b) Proposed, awaiting live evaluation (not wired): one judge call per plan. |
| `Diagnosis.conflict` | `Is<string, "names the tasks or commitments that cannot all hold, with their clock times">` | (b) Proposed, awaiting live evaluation (not wired): one judge call per infeasible request. |
| `Proposal.questions` | `Is<string[], "each a single question the user can answer in a few words">` | (b) Proposed, awaiting live evaluation (not wired). |
| `Proposal` (added) | `Is<Proposal, "...">` | (a) `CheckedProposal`, not in the original table: chosen carries placements and no questions, unclear carries questions and no placements, infeasible neither. The host reads `status`, `placements` and `questions` separately. |

Counts for the 16 candidates: (a) 6, (b) 3, (c) 4, (d) 3. One further (a) row was added (`Proposal`).

The model-facing text added to signatures is the `Is<...>` predicate of `CheckedTaskReading`, `CheckedLimitReading`,
`CheckedPreferenceReading`, `CheckedDomains`, `CheckedPlacements` and `CheckedProposal`. No instruction sentence was
changed.

## Limitations met

See "Language and runtime limitations" at the end of this file (filled in as they were found).

## Changes from today

- 160 lines of crisp TypeScript with one `nl` call become a natural-language scheduler (`scheduler.nl` and 10 stages)
  over a crisp verifier and commit.
- Crisp enumeration stays, as the exact oracle and as one implementation of a pluggable hot path.
- The request can now introduce tasks and commitments, and carries hard limits and weighted preferences.
- Results gain `explanation`, and a status `unclear` that carries the questions to ask.

## Language and runtime limitations

- **Intersection types in `types.ts` are rejected** by the type parser used for `.nl` frontmatter (`bad character at 412:
  " & { id: str"`), so `DayView.fixed` uses a named `Commitment` record instead of `Span & { id: string }`.
- **A function name cannot contain a hyphen** (`"read-limits" is not a valid callable name`), so stages are camelCase.
- **The type of an eval binding is inferred from the nested awaited call.** `const x = await Promise.all(a.flatMap(o =>
  b.map(async w => ({ ..., fit: await assess(o, w) }))))` was refused with `let/x/0: type-mismatch, expected Fit, got
  {...}` (`ts-host/src/native/runtime.ts:1990`, staging of `let` bindings). Instructions spell the form that is accepted:
  build the pairs, `await Promise.all(pairs.map(pair => assess(...)))`, then assemble the records.
- **A for loop must compare its counter with a bound.** `for (...; !found && s + n <= end; ...)` is a `forbidden-loop`
  (`ts-host/src/compiler/policy.ts:247`); `construct.nl` and `repair.nl` therefore state counted loops with a computed
  count.
- **Recursion in eval is limited to a smaller argument**, so `construct.nl` recurses on `remaining` counting down, which
  is harder for a small model to read than an index counting up.
- **A TypeScript dispatcher inside the callable folder works for a pluggable part that is called from natural language**
  (`scheduler/enumerate.ts`, as in `applications/pi`), but not for the entry itself; see the workflow limitations.
