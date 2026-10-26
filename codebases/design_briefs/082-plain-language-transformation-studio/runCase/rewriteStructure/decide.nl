---
description: Decide field-level findings for rewriteStructure.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Reorganize information around reader tasks, use clearer sentences, and introduce explanations or examples where they help. Preserve technical terms when necessary, label examples as illustrative, and avoid creating new rules through oversimplified summaries. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
