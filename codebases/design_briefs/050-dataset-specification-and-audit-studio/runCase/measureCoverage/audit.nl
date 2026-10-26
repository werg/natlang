---
args: { draft: DraftCoverageFinding, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated CoverageFinding field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Exact counts, independent-source accounting, and no false representativeness claim. Explain the decisive evidence and smallest repair target; never weaken the gate.
