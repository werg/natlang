---
description: Decide field-level findings for assembleApplications.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Convert chosen pathways into document, form, appointment, and submission tasks. Check consistency across forms, preserve the applicant’s own statements, and request review for uncertain answers rather than fabricating details to make a form complete. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
