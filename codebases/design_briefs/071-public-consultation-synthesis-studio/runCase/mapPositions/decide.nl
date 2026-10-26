---
description: Decide field-level findings for mapPositions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Group arguments by the actual concern and reasoning, not only topic words. Preserve minority and cross-cutting positions, identify disagreements about facts versus values, and represent tradeoffs without averaging incompatible views into false consensus. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
