---
description: Decide field-level findings for interpretMission.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Translate the mission brief into concrete tasks, capability requirements, and practical constraints. Distinguish essential qualifications from preferences, identify supervision needs, and preserve safety or safeguarding requirements supplied by the organization. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
