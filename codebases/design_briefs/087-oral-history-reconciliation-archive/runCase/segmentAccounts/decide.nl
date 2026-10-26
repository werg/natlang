---
description: Decide field-level findings for segmentAccounts.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Segment interviews into attributable accounts while preserving surrounding context and the speaker’s expressions of uncertainty. Distinguish firsthand recollection, hearsay, later interpretation, and quoted speech rather than presenting them all as equivalent historical evidence. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
