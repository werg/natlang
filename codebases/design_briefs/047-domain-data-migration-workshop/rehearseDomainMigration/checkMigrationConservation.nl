---
args: { records: SourceRecordDisposition[], mappings: DomainFieldMapping[], migration: DomainMigrationRequest }
returns: MigrationConservationReport
---
Compare exact source and target counts by record type, each relationship cardinality, and every stated business conservation invariant. Round-trip representative target values through the mapping and check that reversibility claims hold. Bind all source changes after extraction into a delta set under the frozen mapping version. Fail the rehearsal on any unaccounted row or relationship.
