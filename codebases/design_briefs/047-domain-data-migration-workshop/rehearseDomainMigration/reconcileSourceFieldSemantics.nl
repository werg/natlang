---
args: { migration: DomainMigrationRequest }
returns: SourceMeaningDecision[]
---
For each field reconcile the source dictionary, domain-owner explanation, application context, and representative records. Bind conditional meaning by record type and historical interval. Distinguish null, unknown, not applicable, and withheld as separate source states. If evidence conflicts, leave meaning unresolved instead of inferring semantics from field names or formats.
