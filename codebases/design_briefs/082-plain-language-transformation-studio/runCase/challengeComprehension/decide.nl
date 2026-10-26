---
description: Decide field-level findings for challengeComprehension.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Create realistic comprehension questions and boundary scenarios that test whether readers would act correctly. Include cases involving exceptions and negative requirements, then compare source and rewrite answers to detect semantic drift. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
