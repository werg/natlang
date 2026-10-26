---
description: Decide field-level findings for constructSequence.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Build an exact dependency plan for equipment, access, transport, appointments, and acknowledgments. Identify conditions that must be ready before transfer and expose impossible timing or missing ownership rather than quietly relaxing the approved plan. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
