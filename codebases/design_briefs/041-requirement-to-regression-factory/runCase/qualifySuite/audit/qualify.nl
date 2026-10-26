---
args: { draft: DraftSuiteQualification, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to SuiteQualification: Suite catches required seeded defects and does not pass merely because all tests match current behavior. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
