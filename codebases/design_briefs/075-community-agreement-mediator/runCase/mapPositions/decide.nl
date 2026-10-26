---
description: Decide field-level findings for mapPositions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Summarize each participant’s position in terms they can review and correct. Separate factual disputes, preferences, and nonnegotiable limits, and do not infer hidden motives or psychological explanations from their communication style. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
