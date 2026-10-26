---
description: Decide field-level findings for generateAlternatives.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Invent alternative settings, examples, and interactions that remain plausible for the supplied context. Preserve the same relevant ambiguity, constraints, and skill demand, and avoid making the adapted task easier or changing the intended lesson unintentionally. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
