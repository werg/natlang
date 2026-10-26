---
description: Decide field-level findings for releaseAndTrack.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Assemble a recipient-bound packet, verify exact patient and service identifiers, and obtain required review before transmission. Track delivery, receipt, and administrative acceptance separately, preserving rejection reasons and follow-up obligations. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
