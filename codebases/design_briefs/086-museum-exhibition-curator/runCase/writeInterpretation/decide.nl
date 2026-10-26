---
description: Decide field-level findings for writeInterpretation.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Write labels, introductions, and supporting text that are engaging without overstating evidence. Distinguish fact, attribution, and interpretation, preserve culturally sensitive context, and keep terminology consistent across the visitor journey. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
