---
description: Decide field-level findings for alignConcepts.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare conceptual boundaries across languages and teams, identifying exact, broader, narrower, and partial equivalence. Preserve concepts that do not map cleanly and avoid forcing one language’s categorization onto all others. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
