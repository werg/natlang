# P03 — Build tool with bounded self-repair

Status: first serial execution slice implemented in
[`codebases/build_workbench`](../../codebases/build_workbench/README.md),
[`applications/build_workbench.mjs`](../../applications/build_workbench.mjs), and
the TypeScript host integration test. Cache correctness, repair, teacher quality
and background execution remain open. [Shared capabilities](README.md).

## Natlang prerequisites

C0 is sufficient for serial dependency planning and repair loops; reuse the existing `dependency_plan` pattern. C1 supplies exact process/filesystem operations and long calls. C4 supplies action/output evidence. C5 supports interactive/background builds later. C6 only accelerates independent ready work. A package resolver and execution DAG are library algorithms, not additions to `call` semantics.

## Programme and typed boundary

`build.nl(goal, workspace, policy) -> BuildReport` calls `discover.nl`, `plan.nl`, `choose_ready.nl`, `diagnose.nl`, `propose_repair.nl` and `assess_repair.nl`. Crisp helpers validate the graph, calculate readiness/keys, apply patches and compare outputs. The model chooses semantic dependencies and repair hypotheses; the host executes declared actions.

Tree records describe tasks, dependencies, source digests, commands, expected outputs, completed results and repair attempts. Native workspaces, process objects and cache storage remain in eval. IDs refer to a workspace revision. Completion of an action returns checked summaries, not a raw process object.

## Crisp environment

A retained `ts` host exposes illustrative `workspace.snapshot()`, `build.execute(task)`, `cache.lookup(key)` and `workspace.apply(patch, expectedRevision)`. Native processes may be shared directly inside that environment. A scratch revision isolates a repair from the original workspace. Do not make the initial host implement a universal distributed job protocol.

Cache keys include exact input/transitive dependency identity, toolchain, relevant environment and declared observations. A semantic planning result additionally identifies its model/source/profile. Undeclared file reads either become supported discovered dependencies or make that result uncacheable. A successful cache lookup is verified against stored content identity.

## Reduction and stream shape

Use a bounded Iterate over build state: find ready work, choose/execute it, update the graph state, and stop on success or a precise stall. Do not discard a failed dependency and continue as though the target succeeded. A repair creates a new revision and invalidates affected nodes.

Background mode is a Fold over requested-build, process-result, file-change and cancellation events. Revisions disambiguate results. Parallel execution first uses finite Map over an already validated independent batch; shared cache/publication state is committed through exact ownership rules.

## Delivery and checks

The first slice runs declared argv commands in dependency order. Natlang chooses
ready tasks within the goal's dependency closure. The host rejects stale output,
changed input and invalid paths, records hashes and process status, and labels
an interrupted command's effects unknown. The trace now includes native host
observations even when an eval throws after mutation. This is a trusted command
environment, so caching would currently be unsound: the adapter cannot observe
undeclared reads or all external writes. Enforcing a complete read set, pinning
toolchain/environment identity, and checking cached output content are the next
infrastructure and application gates before enabling reuse. A failed build that
created output requires reconciliation before retry.

1. Build a tiny project serially and compare outputs with a clean reference build.
2. Add caching: no-op reuse, transitive invalidation, changed toolchain and corrupt cache. Gate: cached and clean outputs agree for all exact fixtures.
3. Add one repairable configuration failure and one unrepairable failure. Gate: repair passes original contracts without weakening tests.
4. Add background/concurrent work only after result ownership and revision handling pass.

Use cycles, missing inputs, output collisions, interrupted publication and flaky diagnostics. An uncertain process outcome is inspected/reconciled; it is not blindly retried under a claim of no effects.

## Trace, teacher and portability

Record graph revisions, readiness evidence, cache decisions, command outcomes and patch/validation links. Preserve the original failure. Teacher trajectories train semantic planning and diagnosis, while exact checks establish result correctness. Keep compiler/process requirements in this programme's host closure; small natlang embeddings need not implement them.
