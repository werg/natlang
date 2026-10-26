---
args: { draft: DraftReportingBoundary, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated ReportingBoundary field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Boundary interpretation is explicit and approved before aggregation. Explain the decisive evidence and smallest repair target; never weaken the gate.
