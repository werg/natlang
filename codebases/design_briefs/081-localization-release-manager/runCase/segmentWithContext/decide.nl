---
description: Decide field-level findings for segmentWithContext.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Parse source content into translatable units while preserving structured markup, placeholders, plurals, and surrounding context. Distinguish identical strings used for different purposes and attach the product state or audience needed to choose the correct meaning. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
