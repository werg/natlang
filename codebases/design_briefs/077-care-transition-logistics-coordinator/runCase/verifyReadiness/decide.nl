---
description: Decide field-level findings for verifyReadiness.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Recheck actual delivery, provider acceptance, access, and participant acknowledgments before the transition. Distinguish scheduled from completed prerequisites and record the authorized release decision separately from the program’s readiness summary. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
