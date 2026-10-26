---
description: Decide field-level findings for resolveReferences.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Resolve names, places, and relationships using authorized contextual evidence, preserving aliases and uncertainty. Avoid merging people with similar names or assuming a pronoun’s referent when the interview context supports multiple interpretations. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
