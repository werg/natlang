---
description: Decide field-level findings for coordinateLiveChange.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Obtain producer approval, issue role-specific updates, and confirm that critical staff received the same revision. Track actual execution and adapt only affected elements as new events arrive, preserving an authoritative current run plan. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
