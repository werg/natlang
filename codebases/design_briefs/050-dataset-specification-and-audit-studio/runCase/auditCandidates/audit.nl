---
args: { draft: DraftDatasetRecordAudit, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DatasetRecordAudit field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: No release candidate without verified provenance and permitted intended use. Explain the decisive evidence and smallest repair target; never weaken the gate.
