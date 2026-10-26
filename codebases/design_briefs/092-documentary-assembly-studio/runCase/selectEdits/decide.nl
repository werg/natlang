---
description: Decide field-level findings for selectEdits.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Assemble candidate sequences and transitions under exact duration constraints. Inspect how cuts, adjacency, and narration change the apparent meaning of statements, preserving necessary context and identifying misleading implications even when every individual clip is authentic. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
