---
args: { migration: DomainMigrationRequest }
returns: DomainMigrationProposal
---
1. Bind meanings = reconcileSourceFieldSemantics(migration). A column label or observed format is never sufficient evidence of meaning.
2. Bind mappings = designMeaningPreservingMappings(meanings, migration). Declare domain condition, each missing-state category, and information loss; use system plus validity interval for recycled identifiers.
3. Bind dispositions = classifySourceRecords(mappings, migration). Accept only records satisfying target schema and source meaning. Quarantine valid source cases the target cannot represent instead of fabricating a target value.
4. Bind rehearsal = checkMigrationConservation(dispositions, mappings, migration). Compare counts, relationships, and round trips; capture source deltas against this exact mapping version.
5. Bind cutover = reconcileCutoverBoundary(rehearsal, migration). Freeze mapping hash, replay every delta after extraction, and account for every source record before switch. If rollback conflicts with target writes, create a reconciliation plan, never overwrite new target work.
6. Return { migrationId: migration.migrationId, meanings, mappings, dispositions, rehearsal, cutover, commitAllowed: false }.
