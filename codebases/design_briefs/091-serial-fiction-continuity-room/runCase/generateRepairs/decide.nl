---
description: Decide field-level findings for generateRepairs.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Generate distinct repairs such as changing the new scene, seeding earlier evidence, or revising an approved premise through explicit editorial choice. Preserve voice and emotional intent, and explain narrative tradeoffs rather than selecting the easiest continuity patch. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
