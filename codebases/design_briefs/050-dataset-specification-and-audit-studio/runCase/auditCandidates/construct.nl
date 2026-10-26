---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDatasetRecordAudit
---
Construct a DatasetRecordAudit draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Inspect candidate records for task relevance, data quality, provenance, consent, and leakage risks. Distinguish missing coverage from invalid records, and quarantine material whose rights or source integrity cannot be established.
