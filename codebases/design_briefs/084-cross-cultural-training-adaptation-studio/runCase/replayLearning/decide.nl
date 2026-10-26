---
description: Decide field-level findings for replayLearning.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare original and adapted scenarios through representative correct, incorrect, and borderline responses. Detect shifts in difficulty or meaning, inspect whether examples imply unsupported norms, and revise only the elements responsible for the drift. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
