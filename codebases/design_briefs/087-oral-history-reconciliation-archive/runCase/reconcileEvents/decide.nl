---
description: Decide field-level findings for reconcileEvents.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare accounts of potentially shared events, aligning chronology and scope while preserving different perspectives. Distinguish factual contradiction from differences in emphasis or memory precision, and do not manufacture one authoritative narrative from majority agreement. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
