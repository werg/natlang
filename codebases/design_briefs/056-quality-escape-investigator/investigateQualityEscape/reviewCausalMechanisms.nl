---
args: { trace: EscapeTrace, comparison: DefectRateComparison, histories: ProcessHistory[], evidence: Evidence[] }
returns: CausalReview
---
Bind rivals = formCompetingMechanisms(trace, histories). Bind tests = testRivalMechanismsAgainstEvidence(rivals, comparison, evidence). Keep a person's presence in a history as provenance only; causal attribution requires predicted mechanism evidence and contrary evidence must remain visible.
