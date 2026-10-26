---
description: Decide field-level findings for resolveTerminology.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Match terms to concepts and approved locale guidance, considering polysemy, grammatical role, and product meaning. Preserve scoped exceptions and route genuine terminology conflicts to the owner rather than choosing whichever translation appears most often. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
