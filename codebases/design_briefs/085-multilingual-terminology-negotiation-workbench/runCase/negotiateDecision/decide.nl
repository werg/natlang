---
description: Decide field-level findings for negotiateDecision.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Present alternatives and concrete usage examples to the relevant experts, reconcile their comments, and distinguish stylistic preference from conceptual disagreement. Record scoped acceptance and unresolved dissent rather than treating a majority vote as proof of semantic equivalence. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
