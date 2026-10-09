# Build: decomposition, part by part

A build system takes declared tasks (commands with inputs and outputs), works out what the goal needs, runs it in a
valid order, and says what happened. Every part of that has a decision:

- **fn**: its own natural-language function with a typed contract.
- **inline**: one instruction inside its caller.
- **implicit**: left to the model (rare).
- **crisp**: a TypeScript helper in a callable folder, only for trivially deterministic plumbing.
- **pluggable**: one interface with two implementations, crisp and natural language. The service setting
  `build.implementation(point)` selects one.
- **service**: the outside world, in `BuildWorkspace` (index.ts).

The executors are small models, so each function spells its algorithm out as numbered steps over the named data
structures in types.ts.

## State model

- **Decide on a snapshot, then apply a pure, bounded commit.** A round (`build/step.nl`) reads one `BuildState`,
  produces a `Decision` (data), performs the decision's effect through a service, and hands the decision and the
  effect's result to `commit.settle`, a pure function that returns the next state. The commit appends to lists and sets
  a status. It makes no judgments, and it never runs a command.
- **Effects are data.** A `Decision` is `{kind: run|reuse|stop|reject, task, why}`. The effect (`build.execute`,
  `build.reuse`) is a service call whose result is a `TaskResult`, also data. `commit.admit` checks a decision against
  the state before the effect is performed, so a wrong choice never reaches the workspace.
- **Derived values form a DAG.** `Graph` comes from the declarations. `Plan` (closure, order, cycle) comes from the
  graph and the goal. The ready set comes from the graph, the closure and the attempted list. A choice comes from the
  ready tasks. A validity verdict comes from the task's evidence. A diagnosis comes from a result. Nothing is
  recomputed from a later value, and nothing is stored that a function of the state would give.
- **No while loops.** The build is `step.iterateOn(state)` with `withMeasure` and a stop predicate. The measure is
  `2 * (needed tasks not yet attempted) + (1 when running)`: every round attempts one task or stops, so it strictly
  decreases, and the loop is bounded by the size of the closure.

## Policy

- **Natural language: everything a build system decides.**
  - reading the declarations as a graph (producers, sources, faults);
  - the closure of a goal and a valid order, including cycle detection;
  - which ready task to run next;
  - whether a task's recorded outputs are still valid, and why not;
  - what a failure means and what to change;
  - the closing report.
- **Crisp: only plumbing with one right answer.**
  - the pure commit of a round (append, set status);
  - the guard that the chosen task is ready;
  - the crisp side of each pluggable point.
- **Service: the outside world.**
  - running a command without a shell, with a clean environment, a timeout and bounded output;
  - path containment: paths stay inside the root and out of the host's cache and ledger;
  - input digests before and after, output digests after;
  - refusing preexisting outputs, unless the ledger recorded exactly those bytes;
  - the exact built-in operations (`copy`, `concat`, `uppercase`) and their content-addressed cache;
  - the ledger of last successful runs, written atomically.
- **Teach in errors, not prompts.** Instructions state the wanted behavior. A wrong decision is caught by the exact
  guard, and its message names the rule and the fix (`task goal reads out/a.txt, which task a produces: add a to
  needs of goal`). The model reads that message in the next round's state.

## Parts

| Part | Decision | Unit | Why |
|---|---|---|---|
| Producers: which task writes each path; output clashes | fn (part of declare) | `build/declare` | The graph's first fact. Computed exactly in eval from spelled steps. |
| Derived dependencies: the tasks whose outputs a task reads, merged with `needs` | fn (part of declare) | `build/declare` | A real build system infers them. A declaration that omits one is a fault with the fix in its message. |
| Sources: inputs nobody produces | fn (part of declare) | `build/declare` | Needed by the report and by validity. |
| Declaration faults: empty or duplicate ids, unknown or duplicate needs, self dependency, missing command or output, same output twice, task reading its own output | fn (part of declare) | `build/declare` | Each is a rule with a code and a fix. They are the graph's content, not plumbing. |
| Unknown goal | inline | `build.nl` | One membership test on the nodes. |
| Closure of the goal | fn | `build/closure` | A reachability algorithm over `deps`, spelled as frontier rounds with a counted bound. |
| Order and cycles (Kahn by levels, smaller id first) | fn | `build/order` | Its own algorithm and its own result type. A leftover set is a cycle. |
| Invalid graph or cycle: stop before running anything | inline | `build.nl` | Two status assignments from the plan. |
| Initial state | inline | `build.nl` | An object literal from earlier values. |
| The build loop | inline | `build.nl` | `step.iterateOn(state, files).withMeasure(...).until(...)`. The measure and predicate are spelled. |
| Ready set: needed, not attempted, every dep finished | pluggable | `build/step/ready` | A hot path of every round. Crisp: a filter. Natural language: `ready/rule`. |
| Choose among ready tasks: relevance to the goal, prerequisite descriptions, a named input read from files, smallest id | pluggable | `build/step/choose` | The scheduling policy, and the part that most deserves language. Crisp: smallest id. Natural language: `choose/pick`. |
| Admission: the choice is ready | crisp | `build/step/commit.admit` | An exact guard before an effect, whichever implementation chose. |
| Cache validity: does the ledger entry still match inputs, outputs and declaration | pluggable | `build/step/validity` | The invalidation reasoning, a hot path of every round. Crisp: digest comparison. Natural language: `validity/judge`, which also words the reason. |
| Invalidation propagation | inline | `build/step` | A stale task changes its outputs, so the dependents' inputs differ when their turn comes and their validity says so. The round states this rule once, and the ledger holds the digests. |
| Run a task | service | `build.execute` | A process without a shell, digest checks, containment, cache. |
| Reuse a task | service | `build.reuse` | It re-checks the ledger entry against the files before it reports a task settled. A wrong verdict fails the reuse, and the round runs the task instead. |
| Failure diagnosis from command output and workspace messages | fn | `build/step/diagnose` | A spelled classification of exit codes, missing tools, clashes, mutated inputs, timeouts, with the culprit and the fix. |
| Unknown outcome (timeout, signal) is not retried | fn (part of diagnose) | `build/step/diagnose` | The retry field says `inspect-first`. The service records the same task once per workspace. |
| Commit a round: append to order and results, record judgments, set status and detail | crisp | `build/step/commit.settle` | Pure, bounded bookkeeping. |
| Blocked report: the unfinished closure | crisp (part of settle) | `build/step/commit.settle` | A list difference. |
| Build report: counts, lists, status | inline | `build.nl` | Copies fields of the final state. |
| Build report: summary and next steps | fn | `build/summarize` | Words for a person, from the final state. |
| Ledger of last successful runs, atomic writes | service | `BuildWorkspace` | Durability. |
| Command line | crisp | `main.ts` | The outside world. |

## Pluggable hot paths

| Point | Setting | Crisp implementation | Natural-language implementation | Default |
|---|---|---|---|---|
| `ready` | `build.implementation('ready')` | `step/ready.ts`: a filter over `deps` | `step/ready/rule.nl` | crisp |
| `choose` | `build.implementation('choose')` | `step/choose.ts`: smallest id | `step/choose/pick.nl` | natural-language |
| `validity` | `build.implementation('validity')` | `step/validity.ts`: digest comparison | `step/validity/judge.nl` | natural-language |

The interface of each is one function. `ready(graph, needed, attempted) -> string[]`. `choose(ready, goal, files?) ->
string`. `validity(task, evidence) -> Validity`. The `.ts` file selects. A host picks implementations when it creates
the workspace (`new BuildWorkspace(root, { policy: { choose: 'crisp' } })`) or on the command line
(`--choose crisp`).

## Refinements

Refinement types (`Is<T, "predicate">`, plans/REFINEMENT_TYPES.md). Decisions: (a) adopted with a crisp checker in
`refinements.ts` (no model call), (b) a natural-language judge (proposed only: awaiting live evaluation, not wired),
(c) left to the check that already enforces it exactly (`declare`'s problems, the workspace, `commit`), (d) not
adopted: the value alone does not show the property. The declarations (`Task`) and the state the commit builds keep
plain types. The refined result types are declared as `Checked*` aliases in `types.ts` and named in the `returns` of
the stage that produces the value (`order`, `step/validity/judge`, `step/diagnose`, `build`), so crisp code that
builds the plain `Plan`, `Validity` and `Diagnosis` values is untouched.

| Slot | Proposed type | Decision |
|---|---|---|
| `Task.id` | `Is<string, "non-empty, and different from every other task's id">` | (c) `declare` reports `empty-id` and `duplicate-id`; the declarations are the user's input, not a model's output. |
| `Task.needs` | `Is<string[], "ids of other tasks, without duplicates">` | (c) `declare` (`unknown-need`, `self-dependency`, `duplicate-need`). |
| `Task.argv` | `Is<string[], "a command: ...">` | (c) The workspace refuses an invalid declaration. |
| `Task.inputs`, `Task.outputs` | `Is<string[], "paths relative to the build root that stay inside it">` | (c) The workspace (`path-escape`). |
| `Task.outputs` | `Is<string[], "not empty, and not also among the task's inputs">` | (c) `declare` (`no-output`, `self-read`). |
| `TaskResult.input_sha256`, `.output_sha256` | `Is<string, "64 lowercase hex digits, or empty when the task did not finish">` | (c) Built by the workspace, not by a model. |
| `Plan.needed` | `Is<string[], "the goal and every task any of them depends on, sorted, nothing else">` | (a) Weakened to what the plan shows: sorted without duplicates and containing the goal (`CheckedPlan`). That it is the exact closure needs the graph. |
| `Plan.order` | `Is<string[], "a permutation of needed in which every task comes after all of its deps">` | (a) Weakened to: `order` and `cycle` together hold each needed task exactly once (`CheckedPlan`). The dependency order needs the graph; the ready filter and the commit enforce it. |
| `Plan.cycle` | `Is<string[], "tasks that wait on each other, directly or through others">` | (d) Needs the graph; the partition of `needed` above is the part the value shows. |
| `choose` return (and `Decision.task`) | `Is<string, "the id of one of the supplied ready tasks">` | (c) `commit.admit` refuses it and the round rejects with the ready tasks; the supplied tasks are not in the value. |
| `Validity.reason` | `Is<string, "one short phrase naming the first mismatch: ...">` | (a) `CheckedValidity`: valid exactly when the reason is `up to date`, an invalid reason is one of the four mismatch phrases with a path. |
| `Diagnosis.culprit` | `Is<string, "a path, executable or task id that appears in the task's declaration or its result">` | (d) The declaration and the result are not in the value. |
| `Diagnosis.retry` | `Is<'no' \| 'after-fix' \| 'inspect-first', "inspect-first exactly when the result's status is unknown">` | (a) `CheckedDiagnosis`: the cause is a known one and the retry follows it (inspect-first for interrupted, no for other, after-fix for the rest). The cause is in the value, the status is not. |
| `BuildState.order` | `Is<string[], "tasks of needed, each after its deps, without duplicates">` | (c) Built by `commit.settle`. |
| `BuildReport` | `Is<BuildReport, "status is done only when the goal is in order and every result is ok">` | (a) `CheckedReport`. |
| `Summary.summary` | `Is<string, "states the status and names the failed or blocked task, if any, in at most three sentences">` | (b) Proposed, awaiting live evaluation (not wired): judged once per build, small, with a crisp check of the blank case. |
| `Summary.next` | `Is<string[], "concrete actions a person can take; empty when the build is done">` | (d) "Empty when done" needs the status; concreteness is a preference. |

Counts: (a) 5, (b) 1, (c) 8, (d) 3.

The model-facing text added to signatures is the `Is<...>` predicate of `CheckedPlan`, `CheckedValidity`,
`CheckedDiagnosis` and `CheckedReport`. No instruction sentence was changed.
