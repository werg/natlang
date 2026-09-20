# P13 — Data reconciliation and migration studio

Status: first finite SQLite migration application implemented in
`codebases/data_migration`, `applications/data_migration.py`, and
`scripts/run_data_migration.py`. Live teacher mapping quality remains unmeasured.

## Natlang prerequisites

C0 already supports semantic mapping and conflict assessment; extend the existing reconciliation example. C1 provides table/query/transaction access. C7 is important for consistent NULL, missing, numeric and record conversion across TS/SQL. C4 adds field-level lineage. C5 is only needed for live imports or an interactive review queue. No new entity or database type is required.

## Programme and typed boundary

`reconcile.nl(sources, targetSchema, policy) -> MigrationPlan` calls `map_fields.nl`, `assess_candidate.nl`, `resolve_conflict.nl` and `explain_change.nl`. Crisp helpers prepare candidate pairs, validate constraints and apply accepted patches.

Tree values include schemas, sampled rows, candidate identities, evidence, patches and unresolved conflicts. Large tables/native connections remain in eval. A proposed mapping retains source column/row references and explicit unit conversions. Similar strings are candidate evidence, not sufficient identity proof.

## Crisp environment

Bind snapshot reads, exact joins/aggregates and conditional patch application. SQL can be a selected engine or accessed through TS helpers, depending on whether direct query generation is useful. If both engines are present, define how they access the same snapshot; do not assume live object identity crosses engines.

The host applies a migration transaction with expected source/target revisions. Semantic entity decisions remain in natlang. Shared native access is acceptable, but the published patch must still be attributable; untracked direct writes cannot be represented as a verified transactional migration.

## Reduction and stream shape

First run a finite Map over candidate matches and Fold over accepted patches/review results. Unresolved cases produce review records. Repeated imports use exact source identities and previously committed decisions to avoid duplicate changes. A later live-import Fold treats changed schemas and operator reviews as explicit events.

Avoid a general cross-database atomicity promise. A single database transaction is the first target. Multi-system migration needs a separate recovery contract before it is advertised.

## Delivery and checks

1. Reconcile two small customer/order exports into a checked snapshot. Gate: no dropped rows, exact totals and correct field conversions.
2. Add false-friend identity cases and conflicting facts. Gate: uncertain matches remain unresolved instead of being silently merged.
3. Preview/apply with revision checks and injected transaction failure. Gate: old snapshot remains valid on failure; successful fields retain lineage.
4. Add incremental imports/review events only after finite import behavior is stable.

Test same-name different entities, missing IDs, absent versus null/empty data, changed units, repeated imports and source changes after preview.

## Trace and teacher

The implemented application calls natlang separately for each source schema's
field mapping and for customer identity decisions. The host validates all
selected columns, converts decimal money exactly, requires matching email for
merges, sends missing IDs/email and conflicting names to review, accounts for
every source row, and constructs a frozen preview. Apply verifies both source
and target snapshots, then writes customers, orders, source identity records,
field lineage and the target revision in one SQLite transaction. Reimport of
unchanged orders is recorded as unchanged; changed committed orders are rejected.
The CLI writes the preview before optional apply.

The main semantic risk is that an email match is evidence, not a universal
identity proof. The bounded synthetic case uses that rule deliberately; wider
domains require a source-specific identity policy and independent review data.
The next gate is live teacher runs on held-out schema and identity cases. A
cross-database transaction, automatic resolution of conflicting names, and
arbitrary currency conversion remain outside this first application's contract.

A 20 September 2026 live Bonsai pilot of `map.nl` used the established chat-tool
adapter and quiesced when the model wrote `source_customer_id`, a field absent
from the declared `Mapping` record. The type boundary correctly rejected the
write. The prompt now spells out the exact result shape. A second 120-second
single-leaf pilot did not finish. These are model/tool-use and latency blockers
for accepting teacher samples; they do not justify weakening the type boundary.

Capture source snapshots or exact row observations, semantic decisions, patch sets and transaction outcomes. Use known synthetic mappings and independently checked real examples as references. Score false merges separately from missed matches. A successful SQL transaction validates application, not the semantic identity judgments that generated it. No native connection must become a natlang value.
