---
description: Decide field-level findings for interpretRequirements.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read the receiving service’s supplied administrative requirements and the clinician’s referral instructions. Identify required records, forms, dates, and signatures while preserving clinician-provided priority and avoiding any independent judgment about clinical necessity. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
