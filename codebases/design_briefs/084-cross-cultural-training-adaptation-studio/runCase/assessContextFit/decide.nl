---
description: Decide field-level findings for assessContextFit.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare scenario elements with supplied audience guidance and reviewer input. Identify unfamiliar references, inappropriate assumptions, or practical mismatches without inferring cultural traits from nationality, ethnicity, or other sensitive attributes. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
