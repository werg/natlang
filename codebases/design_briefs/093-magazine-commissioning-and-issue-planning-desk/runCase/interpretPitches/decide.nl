---
description: Decide field-level findings for interpretPitches.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read pitches against the issue brief to identify the actual angle, reporting plan, and audience value. Distinguish an interesting topic from a viable story, preserve uncertainty about access or evidence, and identify overlap with already commissioned work. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
