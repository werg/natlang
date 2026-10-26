---
args: { mappings: DomainFieldMapping[], migration: DomainMigrationRequest }
returns: SourceRecordDisposition[]
---
Apply the approved mapping to bounded record groups. Preserve source identity, relationship provenance, and exception evidence. Accept only target records that pass both schema checks and source-meaning checks. Keep semantic exceptions and unrepresentable variants quarantined with a reason; do not inflate completion counts by merging distinct categories.
Bind identities = qualifyHistoricalIdentityKeys(migration.sourceRecords, mappings). Use source system and validity interval whenever one identifier names different entities over time.
