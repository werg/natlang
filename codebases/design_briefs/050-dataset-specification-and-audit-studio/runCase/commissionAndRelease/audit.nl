---
args: { draft: DraftDatasetRelease, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated DatasetRelease field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Rights, quality, split integrity, and required coverage gates pass or limitations block release. Explain the decisive evidence and smallest repair target; never weaken the gate.
