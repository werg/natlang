# S04 — Changes that preserve intent across artifacts

Status: proposed. Part of [the semantic software plan](../SEMANTIC_SOFTWARE.md).

## Product and semantic ambition

A change should be expressed in terms of the behavior or meaning to preserve,
and natlang should carry it through the affected representations. This includes
reconciling simultaneous changes whose textual patches conflict but whose
intentions may be compatible.

Example: “Represent durations in seconds everywhere, but keep existing imported
millisecond files usable.” Natlang updates schema, import conversion, analyses,
tests, interface labels and explanatory text. Another branch changes the cohort
definition. A semantic merge should preserve both purposes where possible and
show where the combined result needs a decision.

This extends the notional CRDT project. The semantic merge is performed by
natlang. No crisp convergence algebra or universal meaning-preservation rule is
introduced.

## User experience

A change workspace contains the request, inferred intent, candidate edits,
behavioral examples, affected artifacts and outstanding questions. Users inspect
working before/after results alongside diffs. Natlang may carry out authorized
reversible local work without asking about every file.

After completion, the user can inspect “What did this change mean?” and follow
links from the intent to individual edits and verified outcomes. For ambiguous
merges, show concrete alternative behaviors and the evidence each preserves.
Do not force a fluent single answer when the intentions are incompatible.

## Natlang functions

1. `understand_change.nl(request, artifacts) -> Intent`: infer purpose, scope,
   preserved behavior and uncertainties. Author statements and inferred intent
   remain distinguishable.
2. `discover_impact.nl(intent, workspace) -> Work`: follow exact references and
   search for semantic relationships, including prose with no formal dependency.
3. `implement_change.nl(intent, candidate)`: edit actual source/data/view artifacts,
   execute checks and iterate. There is no one-patch-per-event limitation.
4. `evaluate_preservation.nl(before, after, intent, observations)`: judge semantic
   success, inspect unintended changes and extend tests when needed.
5. `merge_intents.nl(base, changes, policy) -> CandidateOrAlternatives`: reconcile
   source histories, purposes and observed behaviors.
6. `explain_change.nl(result)`: produce an evidence-linked account of the outcome.

## Data and exact host responsibilities

`Intent` includes origin, request text, scope, preserved properties, examples and
uncertainties. `ChangeSet` references base manifest, artifact edits, migration
source, evidence, unresolved issues and causal parents. `PreservationClaim` links
a property to actual checks or semantic review, with unverified claims explicit.

The host reads snapshots, calculates diffs/hashes, checks source closure, runs
tests and atomically activates a manifest. Natlang decides what to change and
whether meaning was preserved. Tests are executable evidence, not the complete
definition of intent; a model can otherwise “fix” a failure by weakening a test.
Retain preexisting tests and separately report intentionally changed assertions.

Use ordinary workspace branches first. A repository integration may map them
to isolated worktrees. External effects require explicit operation receipts and
cannot be rolled back by changing a local manifest pointer.

## Semantic replication experiments

Build two execution strategies over the same scenarios:

- **History reduction:** natlang receives a shared base and presented set of
  edits/intents, then derives a candidate combined workspace.
- **State plus update:** natlang interprets each incoming update in context and
  revises current state, preserving enough history to explain its choices.

Use shared root seed and matching model configuration. Pin source, tokenizer,
decoder settings, ordered inputs, evaluator observations and logical seed
derivation. Deduplication, transport order and hashes can be crisp; interpretation
and reconciliation remain semantic. An arrival-order or grouping difference is
an experimental result, not something a hidden crisp merge should conceal.

Two replicas can disagree. Store competing candidates and compare them. A
shared-history recomputation is a supported reconciliation experiment, not a
proof that every incremental sequence converges. Cross-backend repeatability
must be measured rather than promised from seeds alone.

## Delivery sequence

1. One change spanning notebook function, view label, data interpretation and
   explanatory claim. Execute and show all four before/after outcomes.
2. Add semantic impact search and an intentionally indirect dependency so exact
   source imports alone cannot identify the complete change.
3. Add a two-branch merge with compatible intentions and conflicting text.
4. Add incompatible intentions, delete/edit, rename/split and an erroneous
   inferred intent; preserve alternatives and allow correction.
5. Integrate schema migrations from S02 and learned tools from S01, then exercise
   streamed duplicate/late delivery with the existing semantic merge programs.
6. Reuse in executable wiki, repository migration and publishing.

## Tests and teacher tasks

Exact checks: stale bases, missing references, failing generated code, atomic
activation, duplicate updates, incomplete effect receipts and interruption
between patch execution and result recording.

Semantic families: unit changes, terminology with domain-specific exceptions,
API behavior changes, incompatible categorization, prose/code disagreement,
task reprioritization and game-rule evolution. Vary history order and grouping.
Include cases where a tiny text conflict represents a deep semantic conflict,
and large textual changes preserve precisely the same behavior.

Acceptance: the requested behavior works across all affected artifacts; a
compatible independent change remains effective; incompatible requirements
are visible. Independent graders inspect held-out behaviors and preservation
claims, not just diffs or replica equality. Keep unresolved runs in the corpus
when recognizing ambiguity was the correct result.

## Risks

Intent is inferred and can be wrong. Preserve the request and source context,
allow correction and do not treat the model's first paraphrase as authority.
Impact discovery is also fallible: report searched scope and unresolved
references rather than claiming global completeness after local checks.
