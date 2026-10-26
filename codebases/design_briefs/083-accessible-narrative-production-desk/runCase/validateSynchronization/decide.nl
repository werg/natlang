---
description: Decide field-level findings for validateSynchronization.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Run exact timing, overlap, reading-rate, and format checks, then review narrative completeness and usability. Inspect transitions and dense scenes where local compliance may still produce an inaccessible overall experience. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
