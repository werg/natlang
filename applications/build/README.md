# build: a build system in natural language

Declared tasks (an argv command with the files it reads and writes) form a dependency graph. The build system reads
the graph, plans what the goal needs, runs one ready task per round, reuses what is still valid, explains failures,
and reports. Its logic is natural language from end to end. Crisp code is the outside world (the workspace service),
the command line, and the bookkeeping that must be exact. [DECOMPOSITION.md](DECOMPOSITION.md) records the decision
for every part.

```
build.nl                  the build: graph, plan, rounds, report
build/declare.nl            dependency-graph understanding: producers, derived dependencies, sources, faults
build/closure.nl            planning: the tasks the goal needs
build/order.nl              planning: a valid order, and the tasks that wait on each other
build/step.nl               one round: decide from a snapshot, perform one effect, commit
  step/ready                  ready set                          pluggable: crisp or ready/rule.nl
  step/choose                 scheduling policy                  pluggable: crisp or choose/pick.nl
  step/validity               cache validity, invalidation       pluggable: crisp or validity/judge.nl
  step/diagnose.nl            failure diagnosis from the workspace's message and the declaration
  step/commit.ts              the exact guard (admit) and the pure commit (settle)
build/summarize.nl          the report's words: summary and next actions
```

## State model

- **Decide on a snapshot, then apply a pure, bounded commit.** A round reads one `BuildState`. It decides (`Decision`
  is data), performs one effect through the `build` service, and `commit.settle` returns the next state.
- **Effects are data.** `{kind: run|reuse|stop|reject, task, why}` is the decision, `TaskResult` the effect's result.
- **Derived values form a DAG.** `Graph`, then `Plan`, then the ready set, the choice, the validity verdict, the
  diagnosis. Nothing derived is stored twice.
- **No while loops.** The rounds are `step.iterateOn(state)` with a measure (`2 * unattempted + running`), so every
  round makes progress and the loop is bounded by the closure.

## Pluggable hot paths

Each point has one interface and two implementations. The `build.implementation(point)` setting selects. Defaults:
`ready` crisp, `choose` and `validity` natural-language.

```ts
const workspace = await new BuildWorkspace(root, { policy: { choose: 'crisp', validity: 'natural-language' } }).open();
```

## Reuse and invalidation

The workspace keeps a ledger of each task's last successful run in the root (`.natlang-build-ledger/`): declaration
digest, input digests, output digests. A round asks `validity(task, evidence)` whether those still match. A valid
task is settled by `build.reuse`, which checks the ledger against the files again and refuses when they differ, so a
wrong verdict cannot reuse stale files; the round then runs the task. A stale task is run: its recorded outputs are
removed first when they still hold the recorded bytes, and any other preexisting output is refused.

## Running

```sh
natlang run applications/build -- graph.json [--goal ID] [--root DIR] [--choose crisp|natural-language]
```

`graph.json` is `{ "goal": "app", "tasks": [{ "id", "needs", "description", "argv", "inputs", "outputs" }] }`.
Commands run without a shell, in the root, with a clean environment. From code:

```ts
import { BuildWorkspace, buildGoal } from './index.js';
const workspace = await new BuildWorkspace(root).open();
const report = await buildGoal(runtime, workspace, 'app', tasks, openFolder(root).root());
```

`@builtin` `copy`, `concat` and `uppercase` are exact operations with a content-addressed cache; arbitrary commands
are never cached by content.

## Tests

`ts-host/test/build-workbench.test.mjs` runs the stages with a scripted model that answers each stage with the
algorithm its instructions spell out, against real processes: order and reports, failure diagnosis, cycles and
declaration faults, interrupted commands, reuse and invalidation, the pluggable settings, and the exact commit.
