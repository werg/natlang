---
description: Decide field-level findings for followEvidence.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read retrieved records in context, extract relevant evidence, and formulate bounded follow-up leads. Distinguish direct evidence from references to missing sources, avoid circular citation chains, and stop low-value leads under the agreed effort budget. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
