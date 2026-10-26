---
args: { meanings: SourceMeaningDecision[], migration: DomainMigrationRequest }
returns: DomainFieldMapping[]
---
For each supported meaning define source condition, target field, transformation, missing-state mapping, and information-loss declaration. Include source system and validity interval in identity when identifiers are reused. If no target representation preserves a valid source case, propose a schema extension or quarantine route.
Bind missingStates = preserveTypedMissingStates(meanings, migration.targetSchema). Reject a mapping that collapses semantically distinct null categories without an approved loss decision.
