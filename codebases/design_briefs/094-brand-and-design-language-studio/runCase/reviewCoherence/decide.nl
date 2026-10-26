---
description: Decide field-level findings for reviewCoherence.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare the asset collection for conceptual, terminological, and factual coherence. Separate hard defects from subjective creative preference, obtain client review of meaningful alternatives, and version the system and dependent assets together. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
