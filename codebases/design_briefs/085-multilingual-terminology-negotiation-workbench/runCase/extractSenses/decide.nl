---
description: Decide field-level findings for extractSenses.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Collect terminology in actual use across expert documents and translations, preserving grammatical and domain context. Separate different senses of the same word and identify cases where different words denote one concept rather than assuming one-to-one lexical equivalence. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
