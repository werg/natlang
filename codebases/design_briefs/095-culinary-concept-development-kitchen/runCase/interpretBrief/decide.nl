---
description: Decide field-level findings for interpretBrief.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Translate the culinary brief into desired experience, service style, supplied dietary requirements, equipment, and budget constraints. Distinguish preferences from mandatory restrictions and use only verified ingredient and allergen information rather than inferring safety from a dish name. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
