# S02 — Schemas as revisable domain hypotheses

Status: candidate schemas, executable source, branch review and activation work in Inquiry Lab; end-to-end migration and semantic fidelity remain unverified. Part of [the semantic software plan](../SEMANTIC_SOFTWARE.md).

## Product and semantic ambition

The application should be able to discover that its current representation is
wrong for the work. It may split a concept, introduce a relationship, distinguish
observation from interpretation or preserve ambiguity instead of forcing every
new datum into an old category.

Example: a notebook originally groups by `device`. New evidence shows that the
field sometimes denotes physical equipment and sometimes a software profile.
Natlang introduces physical device, configuration and observation records,
migrates what can be justified, and retains ambiguous records for investigation.
This is more demanding than adding an optional field to a JSON object.

## User experience

A “Representation” view shows current concepts, examples and relationships.
When a mismatch matters, natlang demonstrates it with concrete records and
develops a candidate model on a branch. The comparison shows:

- What becomes expressible and what previous distinction is lost or changed.
- How existing records, analyses, interfaces and conclusions would change.
- Which mappings are justified, unresolved or intentionally lossy.
- A working preview using migrated data and source.

Users can explore either version. Activation follows workspace policy; ordinary
reversible evolution can be automatic. An irreversible external migration is a
separate effect with its own outcome, not an implied consequence of previewing.

## Natlang functions and records

`diagnose_representation.nl` finds a task-relevant mismatch using observations
and failed queries. `propose_schema.nl` develops concepts and structural type
source. `classify_records.nl` semantically interprets ambiguous examples.
`write_migration.nl` produces executable transformations. `migrate_consumers.nl`
updates functions, queries, views and claims. `evaluate_migration.nl` compares
behavior and reports whether the new model serves the task better.

Records:

- `SchemaRevision`: type/source manifest, concept descriptions, parent and
  motivating examples.
- `Migration`: input and target revisions, executable source, correspondence
  references, assumptions, loss report and evaluation references.
- `UnresolvedMapping`: original record reference, alternatives and what evidence
  would distinguish them.

Raw evidence remains immutable. An interpreted record links to its origin and
interpretation revision. Do not silently make a semantically uncertain field
look exact merely because the target type requires a value.

## Type strategy and execution

Keep a small stable workspace envelope containing IDs of active schema, data,
program and view manifests. Dynamically generated domain types live inside the
versioned program bundle, checked by the existing type loader. Concrete domain
values are validated against that bundle at child-run boundaries.

This avoids forcing the outer reducer's type to enumerate every future domain.
It also avoids pretending that arbitrary unvalidated JSON is a domain type.
Envelope IDs refer to host-owned validated values; typed child programs operate
on the actual records. Prototype this boundary first: if current loaders cannot
validate the required values, retain the failing case and fix that seam.

No second ontology type system, dependent types or runtime schema keyword is
required initially. Domain descriptions explain meaning; structural types check
shape. Neither is a proof that a semantic mapping is correct.

## Host work and consistency

1. Add candidate workspace manifests grouping schema, data, sources and views.
2. Supply exact snapshot reads, transformation execution, shape validation and
   migration statistics: unmatched inputs, multiplicity and field preservation.
3. Natlang decides whether losses or splits are meaningful; counts alone do not
   settle that judgment.
4. Activate a coherent manifest with a revision compare-and-swap. Readers see
   one version. Concurrent edits cause a branch/rebase, not partial activation.
5. Keep old manifests for restoration. Restoring the active pointer does not
   reverse writes already sent to external systems.
6. Events from old UI revisions carry their schema/view identity. Translate
   through an explicit migration when justified, otherwise preserve the draft
   and ask for a new submission against the current representation.

Begin with local datasets and ordinary ID mappings. Extract migration helpers
only once both notebook and data studio need them. A universal migration engine
would encode too much premature domain structure.

## Delivery sequence

1. One stable-schema notebook with inspectable schema source and raw provenance.
2. Add a meaning-preserving field split on a candidate branch, with real data
   and query migration.
3. Add the device/configuration ambiguity case. Support unresolved records as
   an explicit typed variant and show the consequences in analysis.
4. Migrate generated methods and interactions, replay representative user events
   and compare old/new outputs with natlang interpretation of differences.
5. Handle concurrent edits and a failed migration; activate or retain branches
   without losing raw data, drafts or the prior working application.

## Tests and teacher tasks

Mechanics: broken references, invalid target values, data/source mismatch,
partial writes, stale activation, crash recovery, export/import and unavailable
native datasets. Verify exact preservation only where the migration claims it.

Semantic families: person versus organization, event time versus ingestion time,
price versus currency conversion, physical location versus administrative region,
and observations versus inferred entities. Include irrelevant anomalies where
schema change is unnecessary, plus beneficial revisions that need several steps.

Acceptance: the new representation enables a previously misleading analysis,
updates real consumers, retains ambiguity and can be compared with the previous
version. Merely generating plausible type declarations is insufficient.

## Risks and reuse

Schema churn can make the product unintelligible. Require a concrete mismatch
and show the impact of each adopted revision. A loss report can itself be wrong;
independently compare raw inputs and evaluate held-out queries.

Consumers: data reconciliation, evolving wiki concepts, log taxonomies, game
world concepts, package metadata and cross-file type inference. Application
schemas may evolve while the natlang core and structural type system stay small.
