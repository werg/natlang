---
name: natlang-authoring
description: Author, refactor, review, and validate natlang codebases with typed .nl functions and crisp .ts helpers. Use when implementing semantic algorithms, state reducers, generated programs, or application logic in natlang, including long-running programs intended for small interpreter models.
---

# Author natlang codebases

You are the capable program author. The interpreter may be a small model. Put the algorithm, decomposition, data contracts, and recovery decisions in source so execution does not depend on the interpreter inventing an architecture.

Natlang should own meaningful decisions and control flow: interpretation, planning, semantic merging, iteration, inspection of results, and revision. Crisp helpers supply useful exact computations and host operations. A host that computes the whole workflow before asking natlang to approve it defeats this design. Equally, making the model multiply numbers by hand wastes its context. Choose the boundary deliberately for the task.

## Establish the contract

Locate the natlang checkout or installed host version before editing. If working outside a checkout, use these bundled references and the installed package's declarations; do not assume `../../spec` exists next to this installed skill. Record the chosen source revision and host. Use current source and tests to establish what runs; flag disagreement with prose specifications rather than silently inventing behavior.

Read [language and authoring](references/language.md) when creating or restructuring code. Read [algorithm patterns](references/patterns.md) for loops, stateful applications, semantic reconciliation, and generated methods. Read [verification and diagnosis](references/verification.md) when testing or debugging. The self-contained [review example](assets/review/review.nl) demonstrates a natural-language orchestrator, a semantic leaf, and exact aggregation; copy the whole `assets/review/` directory, including its `types.ts` and companion folder.

## Build a real program

1. Express the required behavior as typed inputs, results, and observable scenarios. Include ambiguous evidence, empty inputs, and partial failure when relevant. Structural types cannot establish semantic correctness.
2. Write the main algorithm in `.nl`: explicit helper calls, order dependencies, branches, iteration state, completion conditions, and what happens when evidence is insufficient. Use prose for semantic judgments whose criteria cannot be reduced to exact rules.
3. Give each helper one coherent responsibility and a precise signature. Bind its lexical dependencies through companion files or `uses`. Keep intermediate data in typed locals. For a large, sparse, read-only namespace such as project files or an artifact catalog, accept an ordinary `Dict<T>` and let the host bind a lazy implementation. Pass that argument explicitly to helpers that need it.
4. Implement crisp `.ts` helpers where exactness, I/O, presentation, or native libraries help. Check the selected executor; a `.ts` extension does not imply Node, npm imports, SQL, shell, or a sandbox. Prefer portable helpers when they suffice.
5. Exercise the actual source through the runtime. Distinguish loader/type tests, scripted interpreter wiring, live model execution, and semantic evaluation. Fix framework defects in the framework when they are the cause.

## Preserve the execution model

- `args` are readable and immutable. Use the persistent typed eval scope for ordinary locals. The final eval expression that matches the declared return type completes the function. Each callee has private state and its own lexical codebase.
- Imported natlang functions and crisp helpers use ordinary awaited positional calls. Values are passed as values; file names and scope variables are never conflated.
- Host-backed lazy dictionaries are model-facing `Dict<T>` values, not a separate tree type or namespace. Inspect them with scope expressions and pass them as ordinary positional arguments to compatible natlang helpers. Their provider and unobserved contents remain host-owned.
- Functions can call checked named helpers through normal imports. Use ordinary loops, array methods, and `Promise.all` for repeated work; crisp helpers own exact operations.
- `eval` runs TypeScript in a persistent scope across turns. An expression matching the declared return type completes the function. Close each substantive instruction line with `mark_lines`; use `report_blocker` for missing information and `report_error` for invalid work.
- Scope values and files are separate. Authorized agents edit the contents of existing files in the fixed `codebase/` overlay with file tools; they cannot create, delete, or move codebase files. Directory reducers receive `project/`, where normal file creation, editing, moving, and deletion are allowed, and retain selected edits through `commit` or `folder.apply`.
- Long productive runs are intended. Do not add arbitrary codebase, nesting, local, turn, or token limits to make a test finish. Explicit deployment budgets are a host policy. Conversation rollover is compatible with long runs: preserve progress in program state.
- Report real blockers and unresolved semantics. Do not return invented success, manufacture receipts, weaken assertions, or turn unknown effect outcomes into failures that are safe to retry.

For embedding, pair this skill with `natlang-integration` when available. Authoring does not require that skill to be installed: the language and testing references here are standalone.
