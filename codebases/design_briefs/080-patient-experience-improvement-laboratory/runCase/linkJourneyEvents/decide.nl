---
description: Decide field-level findings for linkJourneyEvents.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Relate feedback to appointment, contact, and referral events using authorized identifiers and time windows. Distinguish repeated reports of one episode from independent experiences, and preserve uncertain linkage rather than exposing unrelated patient records. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
