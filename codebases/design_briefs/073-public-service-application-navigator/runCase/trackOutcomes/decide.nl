---
description: Decide field-level findings for trackOutcomes.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Record actual submission and agency responses, distinguish receipt from acceptance or eligibility, and formulate next actions for missing evidence or appeal information under supplied procedures. Preserve the agency’s authority and the applicant’s choices. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
