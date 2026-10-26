---
description: Decide field-level findings for formNarratives.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Generate distinct interpretive arguments that connect artifacts meaningfully rather than merely grouping them by topic. Include alternative perspectives and uncertainties, identify missing evidence, and avoid inventing historical continuity to make a more dramatic story. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
