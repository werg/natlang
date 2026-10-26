---
description: Decide field-level findings for qualifyFeasibility.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare proposals with supplied eligibility, resource, and delivery rules, using exact estimates and qualified assessments where necessary. Separate infeasibility from missing information and provide a clear path for revision rather than silently excluding difficult proposals. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
