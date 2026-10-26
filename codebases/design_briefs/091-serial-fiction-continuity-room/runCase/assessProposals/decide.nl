---
description: Decide field-level findings for assessProposals.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Evaluate proposed twists or scenes against chronology, knowledge, motivations, and world rules. Separate genuine contradiction from a reveal that reinterprets prior evidence, and identify what earlier material would need revision to support the new direction. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
