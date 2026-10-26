---
args: { draft: DraftSuiteQualification, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated SuiteQualification field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Suite catches required seeded defects and does not pass merely because all tests match current behavior. Explain the decisive evidence and smallest repair target; never weaken the gate.
