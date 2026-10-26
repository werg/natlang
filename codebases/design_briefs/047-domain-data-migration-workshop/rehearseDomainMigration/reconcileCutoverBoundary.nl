---
args: { rehearsal: MigrationConservationReport, migration: DomainMigrationRequest }
returns: CutoverReconciliation
---
Compare the approved mapping version and source snapshot with the final cutover boundary. Replay every later source delta and rerun conservation checks before consumer switch. If rollback would overwrite target writes, require a record-level reconciliation plan that preserves both sides. Do not switch consumers while any source record, delta, or mapping change is unaccounted.
Bind rollback = reconcileTargetWritesBeforeRollback(migration.targetWriteEvidence, migration.rollbackPolicy). Preserve target-created or changed values before considering any source restore.
