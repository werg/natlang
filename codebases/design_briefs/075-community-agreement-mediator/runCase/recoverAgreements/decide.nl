---
description: Decide field-level findings for recoverAgreements.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Reconstruct prior agreements from records and participant accounts, distinguishing signed or acknowledged terms from expectations and informal habits. Preserve disagreement about what was agreed and avoid treating one participant’s confident recollection as authoritative. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
