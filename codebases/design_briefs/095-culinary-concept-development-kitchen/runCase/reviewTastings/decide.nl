---
description: Decide field-level findings for reviewTastings.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Interpret structured tasting and service rehearsal notes to identify creative and practical improvements. Distinguish subjective preference from repeatable defects, revise the smallest relevant components, and revalidate cost, dietary information, and service dependencies after every change. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
