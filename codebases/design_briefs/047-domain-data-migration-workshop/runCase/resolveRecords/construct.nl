---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftMigrationDecision
---
Construct a MigrationDecision draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Apply mappings to bounded record groups, resolving semantic exceptions through evidence-backed decisions. Preserve identity and relationship provenance, isolate incompatible records, and avoid forcing all source variants into one target category for completion metrics.
