---
description: Decide field-level findings for resolveConflicts.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Propose permitted logistical alternatives when providers or times change, explaining consequences for the approved plan. Escalate any alternative that could alter clinical requirements and keep the transition blocked until the appropriate team resolves the conflict. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
