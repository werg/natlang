# P20 — Repository maintenance and migration agent

Status: isolated migration workbench in `codebases/repository_migration/` and
`applications/repository_migration.mjs`. [Shared capabilities](README.md).

Natlang searches the task manifest and proposes exact old/new patches. The
host rejects ambiguous context, retains immutable candidate revisions, runs
declared checks in a temporary materialization, and returns a reviewable diff
identity without editing the original checkout. Failed candidates can be
patched and checked again. Tests cover a multi-file signature migration,
hidden caller failure, repair, stale context and unchanged source checkout.

The test run exposed a real harness issue: a child `node --test` inherited
`NODE_TEST_CONTEXT` from the parent test process and exited successfully while
skipping its files. The migration host now clears that variable for declared
checks. Broader check provenance, natlang's own loader/type checks as declared
gates, dirty-worktree import and a reviewed worktree publication operation
remain open.

## Natlang prerequisites

C1 exposes native workspace/process objects; C3 checks and runs changed natlang source; C4 records patches and validation evidence. C0 supplies semantic edit planning and bounded repair. C5 is optional for an interactive session. Source edits are data and host operations, not mutations of the currently executing function definition.

## Programme and typed boundary

`migrate.nl(request, snapshot, policy) -> ChangeReport` calls `find_uses.nl`, `plan_change.nl`, `edit_source.nl`, `interpret_checks.nl` and `review_diff.nl`. Exact helpers apply patches against expected revisions and execute the declared checks. Reuse P03/P05/P07 where their interfaces are proven rather than waiting for all of them to be finished.

Tree records contain source spans, revision IDs, patch descriptions, relevant call/signature evidence and test outcomes. Native repositories, indexes and subprocesses remain inside eval. A patch applies to an exact base; failure to match is evidence to replan, not permission for fuzzy replacement.

## Crisp environment

Illustrative `repo.snapshot()`, `repo.search(query)`, `repo.apply(patch, base)`, `runtime.check(source)` and `checks.run(spec)` methods. Use a task-owned checkout or fixture copy. Direct sharing with native Git/workspace objects is permitted in the selected host; it does not bypass the application's requirement to preserve unrelated changes.

Generated code is loaded under its own source revision and execution settings. A child test programme does not automatically inherit every repository-management capability. Publishing/merging a prepared change is an explicit product operation separate from generating a reviewable result.

## Reduction and stream shape

Finite bounded Iterate: inspect, propose patch, apply to task revision, run checks, assess evidence and either finish or revise. Do not modify acceptance tests merely to make a migration pass. A no-change result is valid when the request is already satisfied.

Interactive progress can later arrive through a session Fold; external file changes and process results carry revisions. The active maintenance programme itself remains pinned even as it edits other files. Shared-host mutation observed outside the patch API must be reconciled before claiming an exact diff.

## Delivery and checks

1. Change a helper signature and update all callers in one example codebase. Gate: load/type checks and original scenario behavior pass.
2. Add deliberately hidden callers and stale patches. Gate: correct diagnostics and preserved original workspace.
3. Perform a small library upgrade with a behavioral change requiring adaptation. Gate: stated migration behavior and protected contracts both hold.
4. Add interactive result streams and broader repositories only after source/effect attribution is reliable.

Test dirty workspaces, generated files, false-positive search results, irrelevant passing tests, ambiguous instructions and repair that weakens requirements.

## Trace and teacher

Record before/after source identities, exact patches, searches, commands and check coverage. Teacher references require an independently valid diff, not merely an explanation that sounds plausible. Hold out repository/task families; retain failed attempts and no-change cases. Full replay needs captured source and reset workspace effects. Trace inspection must never apply recorded patches to the user's live repository.
