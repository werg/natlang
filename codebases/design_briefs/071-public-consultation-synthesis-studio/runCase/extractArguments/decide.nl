---
description: Decide field-level findings for extractArguments.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read submissions in context to separate positions, supporting reasons, factual claims, and proposed actions. Preserve conditional support and mixed views within one submission, and avoid reducing nuanced responses to a single positive or negative label. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
