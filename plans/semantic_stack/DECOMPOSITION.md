# Semantic stack curriculum: natlang-native decomposition

## Unit and target

Two independent authored slates provide at least 200 distinct, substantial natural-language programs across at least 15 domains each. A source adapter makes runnable Program/2 cases with typed natural-language roots, callable folders, current-item scoped ad hoc `nl` literals, and typed child outputs. Source cases are generation inputs, not admitted training trajectories. Teacher trajectories pass exact program/oracle checks, source decontamination, semantic review, and corpus registration separately.

## Functions and boundaries

1. **Understand the task contract (root `.nl`).** Read the objective and named data layout. Decide which evidence is germane and what to ask each item. This is natural-language policy, varying by authored program.
2. **Inspect one scoped item (runtime-generated `nl` literal).** The root constructs a typed literal at the point where the item and current accumulated state are known. Its prose uses a short numbered plan: read the item, interpret the relation in the task rule, preserve a quote or path, and return a typed finding. Avoid brittle keyword matching. This is a distinct natural-language call per item.
3. **Revise accumulated state (`iterateOn`).** Carry the typed finding and existing state through an iterative step. The step may create further focused literals to reconcile conflict, fill a missing relation, or evaluate a candidate. The loop terminates on a structural measure: number of uninspected items or explicit finished predicate. No arbitrary global timeout is part of the language contract.
4. **Synthesize and verify (nested named `.nl` helpers).** The root calls the first named stage. Each later stage is in a callable child folder of the preceding stage and executes while that parent call is active. The final stage returns its grounded finding up the stack; the root assembles the declared result. Semantic synthesis is natural language; exact serialization, hashing, source snapshot and oracle checks are crisp. Sibling `.nl` files under the root do not implement a nested call chain: a sibling is outside the active child's callable scope.
5. **Build a case (crisp source adapter).** The adapter checks IDs, types, file paths, nonempty evidence, source identity, split and provenance, then emits Program/2 JSONL. It does not decide a semantic answer or fabricate a teacher trajectory.
6. **Review and generation (existing teacher pipeline).** A generated case goes through source validation, independent exact/semantic checks, held-source decontamination and teacher collection. A passing execution is not automatically admitted as training data.

## Why the crisp pieces exist

JSONL construction, SHA-256 provenance, deterministic source grouping, type/schema validation, source-decontamination and exact file/oracle comparisons require exactness. They do not choose domain policy or replace the natural-language functions. The builder extends existing `curriculumCase` and collection contracts; it does not add a second runtime or a separate evaluation mechanism.

Several program questions may share one evidence world. The builder assigns those questions the same `source_group` and split group from `source.world_group` or `source.split_unit`; their program IDs remain distinct source IDs. This prevents a shared world's evidence from leaking across training and held splits.

## Quality requirements

A catalog entry counts as a distinct program only when its semantic relation, input structure, control flow, or output objective differs substantively. A domain name swap around the same classification loop does not count. Each runnable source must make the decisive evidence visible in inputs, keep oracle facts out of the opening, use scoped arguments instead of dumping the full folder into a child, and expose a result that can be graded. References replay only as source validity evidence; teacher quality and neuralese call topology require separate review.

The two early 3–4-record seed pilots exposed builder defects: the first used helper filenames that did not match the authored names; the second rendered stages as siblings, so a child could not call the next stage. Both pilots are rejected as builder diagnostics, not model negatives or DPO pairs. The corrected builder was preflighted with actual `read_code` and nested runtime calls through the full Astra and Sol chains, then all expanded cases passed exact scripted Program/2 replay. Live execution, semantic answer review, and training admission remain separate gates.
