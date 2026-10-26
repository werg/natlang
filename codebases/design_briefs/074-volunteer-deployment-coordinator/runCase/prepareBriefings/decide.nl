---
description: Decide field-level findings for prepareBriefings.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Generate role-specific briefings with task boundaries, practical logistics, and escalation routes. Share only necessary participant information, distinguish instructions from optional suggestions, and obtain acknowledgment for consequential responsibilities before deployment. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
