---
description: Decide field-level findings for extractExperiences.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Process surveys, complaints, and calls to identify administrative experiences and their context. Separate access or communication issues from clinical conclusions, preserve mixed experiences, and avoid inferring diagnoses or treatment quality from administrative feedback. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
